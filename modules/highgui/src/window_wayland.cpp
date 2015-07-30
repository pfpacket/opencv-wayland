/*
 * OpenCV HighGUI Wayland Backend
 */

#include "precomp.hpp"

#ifndef _WIN32
#if defined (HAVE_WAYLAND)

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <functional>
#include <memory>
#include <system_error>
#include <chrono>
#include <unordered_map>

#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/epoll.h>
#include <linux/input.h>

#include <wayland-client.h>
#include <wayland-client-protocol.h>
#include <wayland-cursor.h>
#include <wayland-util.h>
#include "xdg-shell-client-protocol.h"
#include <xkbcommon/xkbcommon.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

/*                              */
/*  OpenCV highgui internals    */
/*                              */
class cv_wl_display;
class cv_wl_mouse;
class cv_wl_keyboard;
class cv_wl_input;
class cv_wl_buffer;
class cv_wl_cursor;
class cv_wl_cursor_theme;
class cv_wl_widget;
class cv_wl_titlebar;
class cv_wl_viewer;
class cv_wl_trackbar;
class cv_wl_window;
class cv_wl_core;

using std::weak_ptr;
using std::shared_ptr;
using namespace cv::Error;
namespace ch = std::chrono;

extern shared_ptr<cv_wl_core> cv_core;

#define throw_system_error(errmsg, errnum) \
    CV_Error_(StsInternal, ("%s: %s", errmsg, strerror(errnum)));

static int xkb_keysym_to_ascii(xkb_keysym_t keysym)
{
    return (keysym & 0xff);
}

/*
 * From /usr/include/wayland-client-protocol.h
 * @WL_KEYBOARD_KEYMAP_FORMAT_XKB_V1: libxkbcommon compatible; to
 *  determine the xkb keycode, clients must add 8 to the key event keycode
 */
static xkb_keycode_t xkb_keycode_from_raw_keycode(int raw_keycode)
{
    return raw_keycode + 8;
}

static void draw_argb8888(void *d, uint8_t a, uint8_t r, uint8_t g, uint8_t b)
{
    *((uint32_t *)d) = ((a << 24) | (r << 16) | (g << 8) | b);
}

static void write_mat_to_xrgb8888(cv::Mat const& img, void *data)
{
    assert(data != nullptr);
    assert(img.isContinuous());

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            auto p = img.at<cv::Vec3b>(y, x);
            draw_argb8888((char *)data + (y * img.cols + x) * 4, 0x00, p[2], p[1], p[0]);
        }
    }
}

class epoller {
public:
    epoller() : epoll_fd_(epoll_create1(EPOLL_CLOEXEC))
    {
        if (epoll_fd_ < 0)
            throw_system_error("Failed to create epoll fd", errno);
    }

    ~epoller()
    {
        close(epoll_fd_);
    }

    void add(int fd, int events = EPOLLIN)
    {
        this->ctl(EPOLL_CTL_ADD, fd, events);
    }

    void modify(int fd, int events)
    {
        this->ctl(EPOLL_CTL_MOD, fd, events);
    }

    void remove(int fd)
    {
        this->ctl(EPOLL_CTL_DEL, fd, 0);
    }

    void ctl(int op, int fd, int events)
    {
        struct epoll_event event{0, 0};
        event.events = events;
        event.data.fd = fd;
        int ret = epoll_ctl(epoll_fd_, op, fd, &event);
        if (ret < 0)
            throw_system_error("epoll_ctl", errno);
    }

    std::vector<struct epoll_event> wait(int timeout = -1, int max_events = 16)
    {
        std::vector<struct epoll_event> events(max_events);
        int event_num = epoll_wait(epoll_fd_,
            events.data(), events.size(), timeout);
        if (event_num < 0)
            throw_system_error("epoll_wait", errno);
        events.erase(events.begin() + event_num, events.end());
        return events;
    }

private:
    int epoll_fd_;

    epoller(epoller const&) = delete;
    epoller& operator=(epoller const&) = delete;
    epoller(epoller &&) = delete;
    epoller& operator=(epoller &&) = delete;
};

class cv_wl_display {
public:
    cv_wl_display();
    cv_wl_display(std::string const& disp);
    ~cv_wl_display();

    int dispatch();
    int dispatch_pending();
    int flush();
    int roundtrip();

    // int = events, bool = timeout or not
    std::pair<uint32_t, bool> run_once(int timeout);

    struct wl_shm *shm();
    weak_ptr<cv_wl_input> input();
    uint32_t formats() const;
    struct wl_surface *get_surface();
    struct xdg_surface *get_shell_surface(struct wl_surface *surface);

private:
    epoller poller_;
    struct wl_display *display_;
    struct wl_registry *registry_;
    struct wl_registry_listener reg_listener_{
        &handle_reg_global, &handle_reg_remove
    };
    struct wl_compositor *compositor_ = nullptr;
    struct wl_shm *shm_ = nullptr;
    struct wl_shm_listener shm_listener_{&handle_shm_format};
    struct xdg_shell *shell_ = nullptr;
    struct xdg_shell_listener shell_listener_{&handle_shell_ping};
    shared_ptr<cv_wl_input> input_;
    uint32_t formats_ = 0;

    void init(const char *display);
    static void handle_reg_global(void *data, struct wl_registry *reg, uint32_t name, const char *iface, uint32_t version);
    static void handle_reg_remove(void *data, struct wl_registry *wl_registry, uint32_t name);
    static void handle_shm_format(void *data, struct wl_shm *wl_shm, uint32_t format);
    static void handle_shell_ping(void *data, struct xdg_shell *shell, uint32_t serial);
};

class cv_wl_mouse {
public:
    enum button {
        NONE = 0,
        LBUTTON = BTN_LEFT,
        RBUTTON = BTN_RIGHT,
        MBUTTON = BTN_MIDDLE,
    };

    cv_wl_mouse(struct wl_pointer *pointer);
    ~cv_wl_mouse();

    void set_cursor(uint32_t serial, struct wl_surface *surface, int32_t hotspot_x, int32_t hotspot_y);

private:
    struct wl_pointer *pointer_;
    struct wl_pointer_listener pointer_listener_{
        &handle_pointer_enter, &handle_pointer_leave,
        &handle_pointer_motion, &handle_pointer_button,
        &handle_pointer_axis
    };
    cv_wl_window *focus_window_;

    static void handle_pointer_enter(void *data, struct wl_pointer *pointer, uint32_t serial, struct wl_surface *surface, wl_fixed_t sx, wl_fixed_t sy);
    static void handle_pointer_leave(void *data, struct wl_pointer *pointer, uint32_t serial, struct wl_surface *surface);
    static void handle_pointer_motion(void *data, struct wl_pointer *pointer, uint32_t time, wl_fixed_t sx, wl_fixed_t sy);
    static void handle_pointer_button(void *data, struct wl_pointer *wl_pointer, uint32_t serial, uint32_t time, uint32_t button, uint32_t state);
    static void handle_pointer_axis(void *data, struct wl_pointer *wl_pointer, uint32_t time, uint32_t axis, wl_fixed_t value);
};

class cv_wl_keyboard {
public:
    enum {
        MOD_SHIFT_MASK      = 0x01,
        MOD_ALT_MASK        = 0x02,
        MOD_CONTROL_MASK    = 0x04
    };

    cv_wl_keyboard(struct wl_keyboard *keyboard);
    ~cv_wl_keyboard();

    uint32_t get_modifiers() const;
    std::queue<int> get_key_queue();

private:
    struct {
        struct xkb_context *ctx;
        struct xkb_keymap *keymap;
        struct xkb_state *state;
        xkb_mod_mask_t control_mask;
        xkb_mod_mask_t alt_mask;
        xkb_mod_mask_t shift_mask;
    } xkb_{nullptr, nullptr, nullptr, 0, 0, 0};
    struct wl_keyboard *keyboard_ = nullptr;
    struct wl_keyboard_listener keyboard_listener_{
        &handle_kb_keymap, &handle_kb_enter, &handle_kb_leave,
        &handle_kb_key, &handle_kb_modifiers, &handle_kb_repeat
    };
    uint32_t modifiers_;
    std::queue<int> key_queue_;

    static void handle_kb_keymap(void *data, struct wl_keyboard *keyboard, uint32_t format, int fd, uint32_t size);
    static void handle_kb_enter(void *data, struct wl_keyboard *keyboard, uint32_t serial, struct wl_surface *surface, struct wl_array *keys);
    static void handle_kb_leave(void *data, struct wl_keyboard *keyboard, uint32_t serial, struct wl_surface *surface);
    static void handle_kb_key(void *data, struct wl_keyboard *keyboard, uint32_t serial, uint32_t time, uint32_t key, uint32_t state);
    static void handle_kb_modifiers(void *data, struct wl_keyboard *keyboard,
        uint32_t serial, uint32_t mods_depressed, uint32_t mods_latched, uint32_t mods_locked, uint32_t group);
    static void handle_kb_repeat(void *data, struct wl_keyboard *wl_keyboard, int32_t rate, int32_t delay);
};

class cv_wl_input {
public:
    cv_wl_input(struct wl_seat *seat);
    ~cv_wl_input();

    struct wl_seat *seat() const { return seat_; }
    weak_ptr<cv_wl_mouse> mouse();
    weak_ptr<cv_wl_keyboard> keyboard();

private:
    struct wl_seat *seat_;
    struct wl_seat_listener seat_listener_{
        &handle_seat_capabilities, &handle_seat_name
    };
    shared_ptr<cv_wl_mouse> mouse_;
    shared_ptr<cv_wl_keyboard> keyboard_;

    static void handle_seat_capabilities(void *data, struct wl_seat *wl_seat, uint32_t capabilities);
    static void handle_seat_name(void *data, struct wl_seat *wl_seat, const char *name);
};

class cv_wl_buffer {
public:
    cv_wl_buffer();
    ~cv_wl_buffer();

    void destroy();
    void busy(bool busy = true);
    bool is_busy() const;
    cv::Size size() const;
    bool is_allocated() const;
    char *data();
    void create_shm(struct wl_shm *shm, cv::Size size, uint32_t format);
    void attach_to_surface(struct wl_surface *surface, int32_t x, int32_t y);

private:
    int fd_ = -1;
    bool busy_ = false;
    cv::Size size_{0, 0};
    struct wl_buffer *buffer_ = nullptr;
    struct wl_buffer_listener buffer_listener_{
        &handle_buffer_release
    };
    void *shm_data_ = nullptr;

    static int create_tmpfile(std::string const& tmpname);
    static int create_anonymous_file(off_t size);
    static void handle_buffer_release(void *data, struct wl_buffer *buffer);
};

class cv_wl_cursor {
public:
    friend cv_wl_cursor_theme;

    ~cv_wl_cursor();

    std::string const& name() const;
    void set_to_mouse(cv_wl_mouse& mouse, uint32_t serial);
    void commit(int image_index = 0);

private:
    std::string name_;
    struct wl_cursor *cursor_;
    struct wl_surface *surface_;
    struct wl_callback *frame_callback_ = nullptr;
    struct wl_callback_listener frame_listener_{
        &handle_cursor_frame
    };

    cv_wl_cursor(weak_ptr<cv_wl_display> const&, struct wl_cursor *, std::string const&);
    static void handle_cursor_frame(void *data, struct wl_callback *cb, uint32_t time);
};

class cv_wl_cursor_theme {
public:
    cv_wl_cursor_theme(weak_ptr<cv_wl_display> const& display, std::string const& theme, int size = 32);
    ~cv_wl_cursor_theme();

    int size() const;
    std::string const& name() const;
    weak_ptr<cv_wl_cursor> get_cursor(std::string const& name);

private:
    int size_;
    std::string name_;
    weak_ptr<cv_wl_display> display_;
    struct wl_cursor_theme *cursor_theme_ = nullptr;

    std::unordered_map<std::string, shared_ptr<cv_wl_cursor>> cursors_;
};

/*
 * height_for_width widget management
 */
class cv_wl_widget {
public:
    cv_wl_widget(cv_wl_window *window)
        :   window_(window)
    {
    }
    virtual ~cv_wl_widget() = default;

    /* Return the size the last time when we did drawing */
    /* draw method must update the last_size_ */
    virtual cv::Size get_last_size() const
    {
        return last_size_;
    }

    virtual void get_preferred_width(int& minimum, int& natural) const = 0;
    virtual void get_preferred_height_for_width(int width, int& minimum, int& natural) const = 0;

    virtual void on_mouse(int event, cv::Point const& p, int flag) {}

    /* Return: The area widget rendered, if not rendered at all, set as width=height=0 */
    virtual cv::Rect draw(void *data, cv::Size const&, bool force) = 0;

protected:
    cv::Size last_size_{0, 0};
    cv_wl_window *window_;
};

class cv_wl_titlebar : public cv_wl_widget {
public:
    enum {
        btn_width = 24,
        btn_margin = 5,
        btn_max_x = 8,
        btn_max_y = 8,
        titlebar_min_width = btn_width * 3 + btn_margin,
        titlebar_min_height = 24
    };

    cv_wl_titlebar(cv_wl_window *window);

    void get_preferred_width(int& minimum, int& natural) const override;
    void get_preferred_height_for_width(int width, int& minimum, int& natural) const override;
    void on_mouse(int event, cv::Point const& p, int flag) override;

    void calc_button_geometry(cv::Size const& size);
    cv::Rect draw(void *data, cv::Size const& size, bool force) override;

private:
    cv::Mat buf_;
    cv::Rect btn_close_, btn_max_, btn_min_;
    cv::Scalar const line_color_ = CV_RGB(0xff, 0xff, 0xff);
    cv::Scalar const bg_color_ = CV_RGB(0x2d, 0x2d, 0x2d);
    cv::Scalar const border_color_ = CV_RGB(0x53, 0x63, 0x53);

    std::string last_title_;

    struct {
        int face = cv::FONT_HERSHEY_TRIPLEX;
        double scale = 0.4;
        int thickness = 1;
        int baseline = 0;
    } title_;
};

class cv_wl_viewer : public cv_wl_widget {
public:
    enum {
        MOUSE_CALLBACK_MIN_INTERVAL_MILLISEC = 15
    };

    cv_wl_viewer(cv_wl_window *, int flags);

    int get_flags() const { return flags_; }
    void set_image(cv::Mat const& img);
    void set_mouse_callback(CvMouseCallback callback, void *param);

    void get_preferred_width(int& minimum, int& natural) const override;
    void get_preferred_height_for_width(int width, int& minimum, int& natural) const override;
    void on_mouse(int event, cv::Point const& p, int flag) override;
    cv::Rect draw(void *data, cv::Size const&, bool force) override;

private:
    int flags_;
    cv::Mat image_;
    cv::Rect last_img_area_;
    bool image_changed_ = false;

    void *param_ = nullptr;
    CvMouseCallback callback_ = nullptr;
};

class cv_wl_trackbar : public cv_wl_widget {
public:
    cv_wl_trackbar(cv_wl_window *window, std::string const& name,
        int *value, int count, CvTrackbarCallback2 on_change, void *data);

    std::string const& name() const;
    int get_pos() const;
    void set_pos(int value);
    void set_max(int count);

    void get_preferred_width(int& minimum, int& natural) const override;
    void get_preferred_height_for_width(int width, int& minimum, int& natural) const override;
    void on_mouse(int event, cv::Point const& p, int flag) override;
    cv::Rect draw(void *data, cv::Size const& size, bool force) override;

private:
    std::string name_;
    int count_;
    cv::Size size_;

    struct {
        int *value;
        void *data;
        CvTrackbarCallback2 callback;

        void update(int v) { if (value) *value = v; }
        void call(int v) { if (callback) callback(v, data); }
    } on_change_;

    struct {
        cv::Scalar bg = CV_RGB(0xa4, 0xa4, 0xa4);
        cv::Scalar fg = CV_RGB(0xf0, 0xf0, 0xf0);
    } color_;

    struct {
        int fontface = cv::FONT_HERSHEY_COMPLEX_SMALL;
        double fontscale = 0.6;
        int font_thickness = 1;
        cv::Size text_size;
        cv::Point text_orig;

        int margin = 10, thickness = 5;
        cv::Point right, left;
        int length() const { return right.x - left.x; }
    } bar_;

    struct {
        int value = 0;
        int radius = 7;
        cv::Point pos;
        bool drag = false;
    } slider_;

    bool slider_moved_ = true;
    cv::Mat data_;

    void prepare_to_draw();
};

struct cv_wl_mouse_callback {
    bool drag = false;
    cv::Point last{0, 0};
    cv_wl_mouse::button button = cv_wl_mouse::button::NONE;

    void reset()
    {
        drag = false;
        last = cv::Point(0, 0);
        button = cv_wl_mouse::button::NONE;
    }
};

struct cv_wl_window_state {
    cv_wl_window_state()
    {
        reset();
    }

    void reset()
    {
        maximized = fullscreen = resizing = focused = false;
    }

    cv::Size prev_size_{0, 0};
    bool maximized, fullscreen, resizing, focused;
};

class cv_wl_window {
public:
    enum {
        DEFAULT_CURSOR_SIZE = 32
    };

    cv_wl_window(shared_ptr<cv_wl_display> const& display, std::string const& name, int flags);
    ~cv_wl_window();

    cv::Size get_size() const;
    std::string const& get_title() const;
    void set_title(std::string const& title);
    cv_wl_window_state const& state() const;

    void show_image(cv::Mat const& image);

    void create_trackbar(std::string const& name, int *value, int count, CvTrackbarCallback2 on_change, void *userdata);
    weak_ptr<cv_wl_trackbar> get_trackbar(std::string const&) const;

    void mouse_enter(cv::Point const& p, uint32_t serial);
    void mouse_leave();
    void mouse_motion(uint32_t time, cv::Point const& p);
    void mouse_button(uint32_t time, uint32_t button, wl_pointer_button_state state, uint32_t serial);

    void update_cursor(cv::Point const& p, bool grab = false);
    void interactive_move();
    void set_mouse_callback(CvMouseCallback on_mouse, void *param);

    void set_minimized();
    void set_maximized(bool maximize = true);

    void show(cv::Size const& new_size = cv::Size(0, 0));

private:
    cv::Size size_{640, 480};
    std::string title_;

    shared_ptr<cv_wl_display> display_;
    struct wl_surface *surface_;
    struct xdg_surface *shell_surface_;
    struct xdg_surface_listener shsurf_listener{
        &handle_surface_configure, &handle_surface_close
    };
    /* double buffered */
    std::array<cv_wl_buffer, 2> buffers_;

    bool next_frame_ready_ = true;          /* we can now commit a new buffer */
    struct wl_callback *frame_callback_ = nullptr;
    struct wl_callback_listener frame_listener_{
        &handle_frame_callback
    };

    cv_wl_window_state state_;
    struct {
        bool repaint_request = false;  /* we need to redraw as soon as possible (some states are changed) */
        bool resize_request = false;
        cv::Size size{0, 0};
    } pending_;

    shared_ptr<cv_wl_viewer> viewer_;
    std::vector<shared_ptr<cv_wl_widget>> widgets_;
    std::vector<cv::Rect> widget_geometries_;

    cv_wl_mouse_callback on_mouse_;

    uint32_t mouse_enter_serial_;
    uint32_t mouse_button_serial_;
    struct {
        std::string current_name;
        cv_wl_cursor_theme theme;
    } cursor_;

    cv_wl_buffer* next_buffer();
    void commit_buffer(cv_wl_buffer *buffer, cv::Rect const&);
    void deliver_mouse_event(int event, cv::Point const& p, int flag);
    std::tuple<cv::Size, std::vector<cv::Rect>> manage_widget_geometry(cv::Size const& new_size);

    static void handle_surface_configure(void *, struct xdg_surface *, int32_t, int32_t, struct wl_array *, uint32_t);
    static void handle_surface_close(void *data, struct xdg_surface *xdg_surface);
    static void handle_frame_callback(void *data, struct wl_callback *cb, uint32_t time);
};

class cv_wl_core {
public:
    cv_wl_core();
    ~cv_wl_core();

    void init();

    shared_ptr<cv_wl_display> display();
    std::vector<std::string> get_window_names() const;
    shared_ptr<cv_wl_window> get_window(std::string const& name);
    void *get_window_handle(std::string const& name);
    std::string const& get_window_name(void *handle);
    bool create_window(std::string const& name, int flags);
    bool destroy_window(std::string const& name);
    void destroy_all_windows();

private:
    shared_ptr<cv_wl_display> display_;
    std::map<std::string, shared_ptr<cv_wl_window>> windows_;
    std::map<void *, std::string> handles_;
};


/*
 * cv_wl_display implementation
 */
cv_wl_display::cv_wl_display()
{
    init(nullptr);
}

cv_wl_display::cv_wl_display(std::string const& display)
{
    init(display.empty() ? nullptr : display.c_str());
}

cv_wl_display::~cv_wl_display()
{
    wl_shm_destroy(shm_);
    xdg_shell_destroy(shell_);
    wl_compositor_destroy(compositor_);
    wl_registry_destroy(registry_);
    wl_display_flush(display_);
    input_.reset();
    wl_display_disconnect(display_);
}

void cv_wl_display::init(const char *display)
{
    display_ = wl_display_connect(display);
    if (!display_)
        throw_system_error("Could not connect to display", errno);

    registry_ = wl_display_get_registry(display_);
    wl_registry_add_listener(registry_, &reg_listener_, this);
    wl_display_roundtrip(display_);
    if (!compositor_ || !shm_ || !shell_ || !input_)
        CV_Error(StsInternal, "Compositor doesn't have required interfaces");

    wl_display_roundtrip(display_);
    if (!(formats_ & (1 << WL_SHM_FORMAT_XRGB8888)))
        CV_Error(StsInternal, "WL_SHM_FORMAT_XRGB32 not available");

    poller_.add(
        wl_display_get_fd(display_),
        EPOLLIN | EPOLLOUT | EPOLLERR | EPOLLHUP
     );
}

int cv_wl_display::dispatch()
{
    return wl_display_dispatch(display_);
}

int cv_wl_display::dispatch_pending()
{
    return wl_display_dispatch_pending(display_);
}

int cv_wl_display::flush()
{
    return wl_display_flush(display_);
}

int cv_wl_display::roundtrip()
{
    return wl_display_roundtrip(display_);
}

std::pair<uint32_t, bool> cv_wl_display::run_once(int timeout)
{
    // prepare to read events
    this->dispatch_pending();
    int ret = this->flush();
    if (ret < 0 && errno == EAGAIN) {
        poller_.modify(wl_display_get_fd(display_),
            EPOLLIN | EPOLLOUT | EPOLLERR | EPOLLHUP);
    } else if (ret < 0) {
        return std::make_pair(0, false);
    }

    auto events = poller_.wait(timeout);
    if (events.empty())
        return std::make_pair(0, true);

    int events_ = events[0].events;
    if (events_ & EPOLLIN) {
        this->dispatch();
    }

    if (events_ & EPOLLOUT) {
        if (this->flush() == 0) {
            poller_.modify(wl_display_get_fd(display_),
                EPOLLIN | EPOLLERR | EPOLLHUP);
        }
    }
    return std::make_pair(events_, false);
}

struct wl_shm *cv_wl_display::shm()
{
    return shm_;
}

weak_ptr<cv_wl_input> cv_wl_display::input()
{
    return input_;
}

uint32_t cv_wl_display::formats() const
{
    return formats_;
}

struct wl_surface *cv_wl_display::get_surface()
{
    return wl_compositor_create_surface(compositor_);
}

struct xdg_surface *cv_wl_display::get_shell_surface(struct wl_surface *surface)
{
    return xdg_shell_get_xdg_surface(shell_, surface);
}

void cv_wl_display::handle_reg_global(void *data, struct wl_registry *reg, uint32_t name, const char *iface, uint32_t version)
{
    std::string const interface = iface;
    auto *display = reinterpret_cast<cv_wl_display *>(data);

    if (interface == "wl_compositor") {
        display->compositor_ = (struct wl_compositor *)
            wl_registry_bind(reg, name, &wl_compositor_interface, version);
    } else if (interface == "wl_shm") {
        display->shm_ = (struct wl_shm *)
            wl_registry_bind(reg, name, &wl_shm_interface, version);
        wl_shm_add_listener(display->shm_, &display->shm_listener_, display);
    } else if (interface == "xdg_shell") {
        display->shell_ = (struct xdg_shell *)
            wl_registry_bind(reg, name, &xdg_shell_interface, version);
        xdg_shell_use_unstable_version(display->shell_, XDG_SHELL_VERSION_CURRENT);
        xdg_shell_add_listener(display->shell_, &display->shell_listener_, display);
    } else if (interface == "wl_seat") {
        struct wl_seat *seat = (struct wl_seat *)
            wl_registry_bind(reg, name, &wl_seat_interface, version);
        display->input_ = std::make_shared<cv_wl_input>(seat);
    }
}

void cv_wl_display::handle_reg_remove(void *data, struct wl_registry *wl_registry, uint32_t name)
{
}

void cv_wl_display::handle_shm_format(void *data, struct wl_shm *wl_shm, uint32_t format)
{
    auto *display = reinterpret_cast<cv_wl_display *>(data);
    display->formats_ |= (1 << format);
}

void cv_wl_display::handle_shell_ping(void *data, struct xdg_shell *shell, uint32_t serial)
{
    xdg_shell_pong(shell, serial);
}


/*
 * cv_wl_mouse implementation
 */
cv_wl_mouse::cv_wl_mouse(struct wl_pointer *pointer)
    :   pointer_(pointer)
{
    wl_pointer_add_listener(pointer_, &pointer_listener_, this);
}

cv_wl_mouse::~cv_wl_mouse()
{
    wl_pointer_destroy(pointer_);
}

void cv_wl_mouse::set_cursor(uint32_t serial, struct wl_surface *surface, int32_t hotspot_x, int32_t hotspot_y)
{
    wl_pointer_set_cursor(pointer_, serial, surface, hotspot_x, hotspot_y);
}

void cv_wl_mouse::handle_pointer_enter(void *data, struct wl_pointer *pointer,
    uint32_t serial, struct wl_surface *surface, wl_fixed_t sx, wl_fixed_t sy)
{
    int x = wl_fixed_to_int(sx);
    int y = wl_fixed_to_int(sy);
    auto *mouse = reinterpret_cast<cv_wl_mouse *>(data);
    auto *window = reinterpret_cast<cv_wl_window *>(wl_surface_get_user_data(surface));

    mouse->focus_window_ = window;
    mouse->focus_window_->mouse_enter(cv::Point(x, y), serial);
}

void cv_wl_mouse::handle_pointer_leave(void *data,
    struct wl_pointer *pointer, uint32_t serial, struct wl_surface *surface)
{
    auto *mouse = reinterpret_cast<cv_wl_mouse *>(data);

    mouse->focus_window_->mouse_leave();
    mouse->focus_window_ = nullptr;
}

void cv_wl_mouse::handle_pointer_motion(void *data,
    struct wl_pointer *pointer, uint32_t time, wl_fixed_t sx, wl_fixed_t sy)
{
    int x = wl_fixed_to_int(sx);
    int y = wl_fixed_to_int(sy);
    auto *mouse = reinterpret_cast<cv_wl_mouse *>(data);

    mouse->focus_window_->mouse_motion(time, cv::Point(x, y));
}

void cv_wl_mouse::handle_pointer_button(void *data, struct wl_pointer *wl_pointer,
    uint32_t serial, uint32_t time, uint32_t button, uint32_t state)
{
    auto *mouse = reinterpret_cast<cv_wl_mouse *>(data);

    mouse->focus_window_->mouse_button(
        time, button,
        static_cast<wl_pointer_button_state>(state),
        serial
    );
}

void cv_wl_mouse::handle_pointer_axis(void *data, struct wl_pointer *wl_pointer,
    uint32_t time, uint32_t axis, wl_fixed_t value)
{
    /* TODO: Support scroll events */
}


/*
 * cv_wl_keyboard implementation
 */
cv_wl_keyboard::cv_wl_keyboard(struct wl_keyboard *keyboard)
    :   keyboard_(keyboard)
{
    xkb_.ctx = xkb_context_new(XKB_CONTEXT_NO_FLAGS);
    if (!xkb_.ctx)
        CV_Error(StsNoMem, "Failed to create xkb context");
    wl_keyboard_add_listener(keyboard_, &keyboard_listener_, this);
}

cv_wl_keyboard::~cv_wl_keyboard()
{
    if (xkb_.state)
        xkb_state_unref(xkb_.state);
    if (xkb_.keymap)
        xkb_keymap_unref(xkb_.keymap);
    if (xkb_.ctx)
        xkb_context_unref(xkb_.ctx);
    wl_keyboard_destroy(keyboard_);
}

uint32_t cv_wl_keyboard::get_modifiers() const
{
    return modifiers_;
}

std::queue<int> cv_wl_keyboard::get_key_queue()
{
    return std::move(key_queue_);
}

void cv_wl_keyboard::handle_kb_keymap(void *data, struct wl_keyboard *kb, uint32_t format, int fd, uint32_t size)
{
    auto *keyboard = reinterpret_cast<cv_wl_keyboard *>(data);

    try {
        if (format != WL_KEYBOARD_KEYMAP_FORMAT_XKB_V1)
            CV_Error(StsInternal, "XKB_V1 keymap format unavailable");

        char *map_str = (char *)mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
        if (map_str == MAP_FAILED)
            CV_Error(StsInternal, "Failed to mmap keymap");

        keyboard->xkb_.keymap = xkb_keymap_new_from_string(
            keyboard->xkb_.ctx, map_str, XKB_KEYMAP_FORMAT_TEXT_V1, XKB_KEYMAP_COMPILE_NO_FLAGS);
        munmap(map_str, size);
        if (!keyboard->xkb_.keymap)
            CV_Error(StsInternal, "Failed to compile keymap");

        keyboard->xkb_.state = xkb_state_new(keyboard->xkb_.keymap);
        if (!keyboard->xkb_.state)
            CV_Error(StsNoMem, "Failed to create XKB state");

        keyboard->xkb_.control_mask =
            1 << xkb_keymap_mod_get_index(keyboard->xkb_.keymap, "Control");
        keyboard->xkb_.alt_mask =
            1 << xkb_keymap_mod_get_index(keyboard->xkb_.keymap, "Mod1");
        keyboard->xkb_.shift_mask =
            1 << xkb_keymap_mod_get_index(keyboard->xkb_.keymap, "Shift");
    } catch (std::exception& e) {
        if (keyboard->xkb_.keymap)
            xkb_keymap_unref(keyboard->xkb_.keymap);
        std::cerr << "OpenCV Error: " << e.what() << std::endl;
    }

    close(fd);
}

void cv_wl_keyboard::handle_kb_enter(void *data, struct wl_keyboard *keyboard, uint32_t serial, struct wl_surface *surface, struct wl_array *keys)
{
}

void cv_wl_keyboard::handle_kb_leave(void *data, struct wl_keyboard *keyboard, uint32_t serial, struct wl_surface *surface)
{
}

void cv_wl_keyboard::handle_kb_key(void *data, struct wl_keyboard *keyboard, uint32_t serial, uint32_t time, uint32_t key, uint32_t state)
{
    auto *kb = reinterpret_cast<cv_wl_keyboard *>(data);
    xkb_keycode_t keycode = xkb_keycode_from_raw_keycode(key);

    if (state == WL_KEYBOARD_KEY_STATE_RELEASED) {
        xkb_keysym_t keysym = xkb_state_key_get_one_sym(kb->xkb_.state, keycode);
        kb->key_queue_.push(xkb_keysym_to_ascii(keysym));
    }
}

void cv_wl_keyboard::handle_kb_modifiers(void *data, struct wl_keyboard *keyboard,
                        uint32_t serial, uint32_t mods_depressed,
                        uint32_t mods_latched, uint32_t mods_locked,
                        uint32_t group)
{
    auto *kb = reinterpret_cast<cv_wl_keyboard *>(data);

    if (!kb->xkb_.keymap)
        return;

    xkb_state_update_mask(
        kb->xkb_.state, mods_depressed,
        mods_latched, mods_locked, 0, 0, group
    );

    xkb_mod_mask_t mask = xkb_state_serialize_mods(
        kb->xkb_.state,
        static_cast<xkb_state_component>(
            XKB_STATE_MODS_DEPRESSED | XKB_STATE_MODS_LATCHED)
    );

    kb->modifiers_ = 0;
    if (mask & kb->xkb_.control_mask)
        kb->modifiers_ |= cv_wl_keyboard::MOD_CONTROL_MASK;
    if (mask & kb->xkb_.alt_mask)
        kb->modifiers_ |= cv_wl_keyboard::MOD_ALT_MASK;
    if (mask & kb->xkb_.shift_mask)
        kb->modifiers_ |= cv_wl_keyboard::MOD_SHIFT_MASK;
}

void cv_wl_keyboard::handle_kb_repeat(void *data, struct wl_keyboard *wl_keyboard, int32_t rate, int32_t delay)
{
}


/*
 * cv_wl_input implementation
 */
cv_wl_input::cv_wl_input(struct wl_seat *seat)
    :   seat_(seat)
{
    wl_seat_add_listener(seat_, &seat_listener_, this);
}

cv_wl_input::~cv_wl_input()
{
    mouse_.reset();
    keyboard_.reset();
    wl_seat_destroy(seat_);
}

weak_ptr<cv_wl_mouse> cv_wl_input::mouse()
{
    return mouse_;
}

weak_ptr<cv_wl_keyboard> cv_wl_input::keyboard()
{
    return keyboard_;
}

void cv_wl_input::handle_seat_capabilities(void *data, struct wl_seat *wl_seat, uint32_t caps)
{
    auto *input = reinterpret_cast<cv_wl_input *>(data);

    if (caps & WL_SEAT_CAPABILITY_POINTER) {
        struct wl_pointer *pointer = wl_seat_get_pointer(input->seat_);
        input->mouse_ = std::make_shared<cv_wl_mouse>(pointer);
    }

    if (caps & WL_SEAT_GET_KEYBOARD) {
        struct wl_keyboard *keyboard = wl_seat_get_keyboard(input->seat_);
        input->keyboard_ = std::make_shared<cv_wl_keyboard>(keyboard);
    }
}

void cv_wl_input::handle_seat_name(void *data, struct wl_seat *wl_seat, const char *name)
{
}


/*
 * cv_wl_buffer implementation
 */
cv_wl_buffer::cv_wl_buffer()
{
}

cv_wl_buffer::~cv_wl_buffer()
{
    this->destroy();
}

void cv_wl_buffer::destroy()
{
    if (buffer_) {
        wl_buffer_destroy(buffer_);
        buffer_ = nullptr;
    }

    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }

    if (shm_data_ && shm_data_ != MAP_FAILED) {
        munmap(shm_data_, size_.area() * 4);
        shm_data_ = nullptr;
    }

    size_.width = size_.height = 0;
}

void cv_wl_buffer::busy(bool busy)
{
    busy_ = busy;
}

bool cv_wl_buffer::is_busy() const
{
    return busy_;
}

cv::Size cv_wl_buffer::size() const
{
    return size_;
}

bool cv_wl_buffer::is_allocated() const
{
    return buffer_ && shm_data_;
}

char *cv_wl_buffer::data()
{
    return (char *)shm_data_;
}

void cv_wl_buffer::create_shm(struct wl_shm *shm, cv::Size size, uint32_t format)
{
    this->destroy();

    size_ = size;
    int stride = size_.width * 4;
    int buffer_size = stride * size_.height;

    fd_ = this->create_anonymous_file(buffer_size);

    shm_data_ = mmap(NULL, buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (shm_data_ == MAP_FAILED) {
        int errno_ = errno;
        this->destroy();
        throw_system_error("failed to map shm", errno_);
    }

    struct wl_shm_pool *pool = wl_shm_create_pool(shm, fd_, buffer_size);
    buffer_ = wl_shm_pool_create_buffer(pool, 0, size_.width, size_.height, stride, format);
    wl_buffer_add_listener(buffer_, &buffer_listener_, this);
    wl_shm_pool_destroy(pool);
}

void cv_wl_buffer::attach_to_surface(struct wl_surface *surface, int32_t x, int32_t y)
{
    wl_surface_attach(surface, buffer_, x, y);
    this->busy();
}

int cv_wl_buffer::create_tmpfile(std::string const& tmpname)
{
    std::vector<char> filename(tmpname.begin(), tmpname.end());
    filename.push_back('\0');

    int fd = mkostemp(filename.data(), O_CLOEXEC);
    if (fd >= 0)
        unlink(filename.data());
    else
        CV_Error_(StsInternal,
            ("Failed to create a tmp file: %s: %s",
                tmpname.c_str(), strerror(errno)));
    return fd;
}

int cv_wl_buffer::create_anonymous_file(off_t size)
{
    auto path = getenv("XDG_RUNTIME_DIR") + std::string("/opencv-shared-XXXXXX");
    int fd = create_tmpfile(path);

    int ret = posix_fallocate(fd, 0, size);
    if (ret != 0) {
        close(fd);
        throw_system_error("Failed to fallocate shm", errno);
    }

    return fd;
}

void cv_wl_buffer::handle_buffer_release(void *data, struct wl_buffer *buffer)
{
    auto *cvbuf = reinterpret_cast<cv_wl_buffer *>(data);

    cvbuf->busy(false);
}


/*
 * cv_wl_cursor implementation
 */
cv_wl_cursor::cv_wl_cursor(weak_ptr<cv_wl_display> const& display, struct wl_cursor *cursor, std::string const& name)
    : name_(name), cursor_(cursor)
{
    surface_ = display.lock()->get_surface();
}

cv_wl_cursor::~cv_wl_cursor()
{
    if (frame_callback_)
        wl_callback_destroy(frame_callback_);
    wl_surface_destroy(surface_);
}

std::string const& cv_wl_cursor::name() const
{
    return name_;
}

void cv_wl_cursor::set_to_mouse(cv_wl_mouse& mouse, uint32_t serial)
{
    auto *cursor_img = cursor_->images[0];
    mouse.set_cursor(serial, surface_, cursor_img->hotspot_x, cursor_img->hotspot_y);
}

void cv_wl_cursor::commit(int image_index)
{
    auto *cursor_img = cursor_->images[image_index];
    auto *cursor_buffer = wl_cursor_image_get_buffer(cursor_img);

    if (cursor_->image_count > 1) {
        if (frame_callback_)
            wl_callback_destroy(frame_callback_);

        frame_callback_ = wl_surface_frame(surface_);
        wl_callback_add_listener(frame_callback_, &frame_listener_, this);
    }

    wl_surface_attach(surface_, cursor_buffer, 0, 0);
    wl_surface_damage(surface_, 0, 0, cursor_img->width, cursor_img->height);
    wl_surface_commit(surface_);
}

void cv_wl_cursor::handle_cursor_frame(void *data, struct wl_callback *cb, uint32_t time)
{
    auto *cursor = (struct cv_wl_cursor *)data;
    int image_index = wl_cursor_frame(cursor->cursor_, time);

    cursor->commit(image_index);
}


/*
 * cv_wl_cursor_theme implementation
 */
cv_wl_cursor_theme::cv_wl_cursor_theme(weak_ptr<cv_wl_display> const& display, std::string const& theme, int size)
    : size_(size), name_(theme), display_(display), cursors_()
{
    cursor_theme_ = wl_cursor_theme_load(theme.c_str(), size, display.lock()->shm());
    if (!cursor_theme_)
        CV_Error_(StsInternal, ("Couldn't load cursor theme: %s", theme.c_str()));
}

cv_wl_cursor_theme::~cv_wl_cursor_theme()
{
    if (cursor_theme_)
        wl_cursor_theme_destroy(cursor_theme_);
}

int cv_wl_cursor_theme::size() const
{
    return size_;
}

std::string const& cv_wl_cursor_theme::name() const
{
    return name_;
}

weak_ptr<cv_wl_cursor> cv_wl_cursor_theme::get_cursor(std::string const& name)
{
    if (cursors_.count(name) == 1)
        return cursors_[name];

    auto *wlcursor = wl_cursor_theme_get_cursor(cursor_theme_, name.c_str());
    if (!wlcursor)
        CV_Error_(StsInternal, ("Couldn't load cursor: %s", name.c_str()));

    auto cursor =
        shared_ptr<cv_wl_cursor>(new cv_wl_cursor(display_, wlcursor, name));
    if (!cursor)
        CV_Error_(StsInternal, ("Couldn't allocate memory for cursor: %s", name.c_str()));

    cursors_[name] = cursor;

    return cursor;
}


/*
 * cv_wl_titlebar implementation
 */
cv_wl_titlebar::cv_wl_titlebar(cv_wl_window *window) : cv_wl_widget(window)
{
}

void cv_wl_titlebar::get_preferred_width(int& minimum, int& natural) const
{
    minimum = natural = titlebar_min_width;
}

void cv_wl_titlebar::get_preferred_height_for_width(int width, int& minimum, int& natural) const
{
    minimum = natural = titlebar_min_height;
}

void cv_wl_titlebar::on_mouse(int event, cv::Point const& p, int flag)
{
    switch (event) {
    case cv::EVENT_LBUTTONDOWN:
        if (btn_close_.contains(p)) {
            exit(EXIT_SUCCESS);
        } else if (btn_max_.contains(p)) {
            window_->set_maximized(!window_->state().maximized);
        } else if (btn_min_.contains(p)) {
            window_->set_minimized();
        } else {
            window_->update_cursor(p, true);
            window_->interactive_move();
        }
    }
}

void cv_wl_titlebar::calc_button_geometry(cv::Size const& size)
{
    /* Basic button geoemetries */
    cv::Size btn_size = cv::Size(btn_width, size.height);
    btn_close_ = cv::Rect(cv::Point(size.width - 5 - btn_size.width, 0), btn_size);
    btn_max_ = cv::Rect(cv::Point(btn_close_.x - btn_size.width, 0), btn_size);
    btn_min_ = cv::Rect(cv::Point(btn_max_.x - btn_size.width, 0), btn_size);
}

cv::Rect cv_wl_titlebar::draw(void *data, cv::Size const& size, bool force)
{
    auto damage = cv::Rect(0, 0, 0, 0);

    if (force || last_size_ != size || last_title_ != window_->get_title()) {
        buf_ = cv::Mat(size, CV_8UC3, bg_color_);
        this->calc_button_geometry(size);

        auto const margin = cv::Point(btn_max_x, btn_max_y);
        auto const btn_cls = cv::Rect(btn_close_.tl() + margin, btn_close_.br() - margin);
        auto const btn_max = cv::Rect(btn_max_.tl() + margin, btn_max_.br() - margin);
        auto title_area = cv::Rect(0, 0, size.width - titlebar_min_width, size.height);

        auto text = cv::getTextSize(window_->get_title(), title_.face, title_.scale, title_.thickness, &title_.baseline);
        if (text.area() <= title_area.area()) {
            auto origin = cv::Point(0, (size.height + text.height) / 2);
            origin.x = ((title_area.width >= (size.width + text.width) / 2) ?
                (size.width - text.width) / 2 : (title_area.width - text.width) / 2);
            cv::putText(
                buf_, window_->get_title(),
                origin, title_.face, title_.scale,
                CV_RGB(0xff, 0xff, 0xff), title_.thickness, CV_AA
            );
        }

        buf_(cv::Rect(btn_min_.tl(), cv::Size(titlebar_min_width, size.height))) = bg_color_;
        cv::line(buf_, btn_cls.tl(), btn_cls.br(), line_color_, 1, CV_AA);
        cv::line(buf_, btn_cls.tl() + cv::Point(btn_cls.width, 0), btn_cls.br() - cv::Point(btn_cls.width, 0), line_color_, 1, CV_AA);
        cv::rectangle(buf_, btn_max.tl(), btn_max.br(), line_color_, 1, CV_AA);
        cv::line(buf_, cv::Point(btn_min_.x + 8, btn_min_.height / 2), cv::Point(btn_min_.x + btn_min_.width - 8, btn_min_.height / 2), line_color_, 1, CV_AA);
        cv::line(buf_, cv::Point(0, 0), cv::Point(buf_.size().width, 0), border_color_, 1, CV_AA);

        write_mat_to_xrgb8888(buf_, data);
        last_size_ = size;
        last_title_ = window_->get_title();
        damage = cv::Rect(cv::Point(0, 0), size);
    }

    return damage;
}


/*
 * cv_wl_viewer implementation
 */
cv_wl_viewer::cv_wl_viewer(cv_wl_window *window, int flags)
    : cv_wl_widget(window), flags_(flags)
{
}

void cv_wl_viewer::set_image(cv::Mat const& image)
{
    if (image.type() == CV_8UC1) {
        cv::Mat bgr;
        cv::cvtColor(image, bgr, CV_GRAY2BGR);
        image_ = bgr.clone();
    } else {
        image_ = image.clone();
    }
    image_changed_ = true;
}

void cv_wl_viewer::set_mouse_callback(CvMouseCallback callback, void *param)
{
    param_ = param;
    callback_ = callback;
}

void cv_wl_viewer::get_preferred_width(int& minimum, int& natural) const
{
    if (image_.size().area() == 0) {
        minimum = natural = 0;
    } else {
        natural = image_.size().width;
        minimum = (flags_ == cv::WINDOW_AUTOSIZE ? natural : 0);
    }
}

static double aspect_ratio(cv::Size const& size)
{
    return (double)size.height / (double)size.width;
}

void cv_wl_viewer::get_preferred_height_for_width(int width, int& minimum, int& natural) const
{
    if (image_.size().area() == 0) {
        minimum = natural = 0;
    } else if (flags_ == cv::WINDOW_AUTOSIZE) {
        assert(width == image_.size().width);
        minimum = natural = image_.size().height;
    } else {
        natural = width * aspect_ratio(image_.size());
        minimum = (flags_ & CV_WINDOW_FREERATIO ? 0 : natural);
    }
}

void cv_wl_viewer::on_mouse(int event, cv::Point const& p, int flag)
{
    // Make sure the first mouse event is delivered to clients
    static int last_event = ~event, last_flag = ~flag;
    static auto last_event_time = ch::steady_clock::now();

    if (callback_) {
        auto now = ch::steady_clock::now();
        auto elapsed = ch::duration_cast<ch::milliseconds>(now - last_event_time);

        /* Inhibit the too frequent mouse callback due to the heavy load */
        if (event != last_event || flag != last_flag ||
            elapsed.count() >= MOUSE_CALLBACK_MIN_INTERVAL_MILLISEC) {
            last_event = event;
            last_flag = flag;
            last_event_time = now;

            /* Scale the coordinate to match the client's image coordinate */
            int x = (p.x - last_img_area_.x) * ((double)image_.size().width / last_img_area_.width);
            int y = (p.y - last_img_area_.y) * ((double)image_.size().height / last_img_area_.height);
            callback_(event, x, y, flag, param_);
        }
    }
}

cv::Rect cv_wl_viewer::draw(void *data, cv::Size const& size, bool force)
{
    if ((!force && !image_changed_ && last_size_ == size) || image_.size().area() == 0 || size.area() == 0)
        return cv::Rect(0, 0, 0, 0);

    last_img_area_ = cv::Rect(cv::Point(0, 0), size);

    if (flags_ == cv::WINDOW_AUTOSIZE || image_.size() == size) {
        assert(image_.size() == size);
        write_mat_to_xrgb8888(image_, data);
    } else {
        if (flags_ & CV_WINDOW_FREERATIO) {
            cv::Mat resized;
            cv::resize(image_, resized, size);
            write_mat_to_xrgb8888(resized, data);
        } else /* CV_WINDOW_KEEPRATIO */ {
            auto rect = cv::Rect(cv::Point(0, 0), size);
            if (aspect_ratio(size) >= aspect_ratio(image_.size())) {
                rect.height = image_.size().height * ((double)rect.width / image_.size().width);
            } else {
                rect.height = size.height;
                rect.width = image_.size().width * ((double)rect.height / image_.size().height);
            }
            rect.x = (size.width - rect.width) / 2;
            rect.y = (size.height - rect.height) / 2;

            auto buf = cv::Mat(size, image_.type(), CV_RGB(0xa4, 0xa4, 0xa4));
            auto resized = buf(rect);
            cv::resize(image_, resized, rect.size());
            write_mat_to_xrgb8888(buf, data);

            last_img_area_ = rect;
        }
    }

    last_size_ = size;
    image_changed_ = false;

    return cv::Rect(cv::Point(0, 0), size);
}


/*
 * cv_wl_trackbar implementation
 */
cv_wl_trackbar::cv_wl_trackbar(cv_wl_window *window, std::string const& name,
    int *value, int count, CvTrackbarCallback2 on_change, void *data)
    :   cv_wl_widget(window), name_(name), count_(count)
{
    on_change_.value = value;
    on_change_.data = data;
    on_change_.callback = on_change;
}

std::string const& cv_wl_trackbar::name() const
{
    return name_;
}

int cv_wl_trackbar::get_pos() const
{
    return slider_.value;
}

void cv_wl_trackbar::set_pos(int value)
{
    if (0 <= value && value <= count_) {
        slider_.value = value;
        slider_moved_ = true;
        window_->show();
    }
}

void cv_wl_trackbar::set_max(int maxval)
{
    count_ = maxval;
    if (!(0 <= slider_.value && slider_.value <= count_)) {
        slider_.value = maxval;
        slider_moved_ = true;
        window_->show();
    }
}

void cv_wl_trackbar::get_preferred_width(int& minimum, int& natural) const
{
    minimum = natural =  320;
}

void cv_wl_trackbar::get_preferred_height_for_width(int width, int& minimum, int& natural) const
{
    minimum = natural = 40;
}

void cv_wl_trackbar::prepare_to_draw()
{
    bar_.text_size = cv::getTextSize(
        (name_ + ": " + std::to_string(count_)).c_str(), bar_.fontface,
        bar_.fontscale, bar_.font_thickness, nullptr);
    bar_.text_orig = cv::Point(2, (size_.height + bar_.text_size.height) / 2);
    bar_.left = cv::Point(bar_.text_size.width + 10, size_.height / 2);
    bar_.right = cv::Point(size_.width - bar_.margin - 1, size_.height / 2);

    int slider_pos_x = ((double)bar_.length() / count_ * slider_.value);
    slider_.pos = cv::Point(bar_.left.x + slider_pos_x, bar_.left.y);
}

cv::Rect cv_wl_trackbar::draw(void *data, cv::Size const& size, bool force)
{
    auto damage = cv::Rect(0, 0, 0, 0);

    if (slider_moved_) {
        on_change_.update(slider_.value);
        on_change_.call(slider_.value);
    }

    if (slider_moved_ || force) {
        size_ = last_size_ = size;

        if (size_ == data_.size())
            data_ = CV_RGB(0xde, 0xde, 0xde);
        else
            data_ = cv::Mat(size_, CV_8UC3, CV_RGB(0xde, 0xde, 0xde));

        this->prepare_to_draw();
        cv::putText(
            data_,
            (name_ + ": " + std::to_string(slider_.value)),
            bar_.text_orig, bar_.fontface, bar_.fontscale,
            CV_RGB(0x00, 0x00, 0x00), bar_.font_thickness, CV_AA);

        cv::line(data_, bar_.left, bar_.right, color_.bg, bar_.thickness + 3, CV_AA);
        cv::line(data_, bar_.left, bar_.right, color_.fg, bar_.thickness, CV_AA);
        cv::circle(data_, slider_.pos, slider_.radius, color_.fg, -1, CV_AA);
        cv::circle(data_, slider_.pos, slider_.radius, color_.bg, 1, CV_AA);

        write_mat_to_xrgb8888(data_, data);
        damage = cv::Rect(cv::Point(0, 0), size);
        slider_moved_ = false;
    }

    return damage;
}

void cv_wl_trackbar::on_mouse(int event, cv::Point const& p, int flag)
{
    switch (event) {
    case cv::EVENT_LBUTTONDOWN:
        slider_.drag = true;
        window_->update_cursor(p, true);
        break;
    case cv::EVENT_MOUSEMOVE:
        if (!(flag & cv::EVENT_FLAG_LBUTTON))
            break;
    case cv::EVENT_LBUTTONUP:
        if (slider_.drag && bar_.left.x <= p.x && p.x <= bar_.right.x) {
            slider_.value = (double)(p.x - bar_.left.x) / bar_.length() * count_;
            slider_moved_ = true;
            window_->show();
            slider_.drag = (event != cv::EVENT_LBUTTONUP);
        }
    }
}


/*
 * cv_wl_window implementation
 */
cv_wl_window::cv_wl_window(shared_ptr<cv_wl_display> const& display, std::string const& title, int flags)
    :   title_(title), display_(display),
        surface_(display->get_surface()),
        cursor_{{}, {display, "default", DEFAULT_CURSOR_SIZE}}
{
    shell_surface_ = display->get_shell_surface(surface_);
    if (!shell_surface_)
        CV_Error(StsInternal, "Failed to get xdg_surface");

    xdg_surface_add_listener(shell_surface_, &shsurf_listener, this);
    xdg_surface_set_title(shell_surface_, title_.c_str());

    wl_surface_set_user_data(surface_, this);

    widgets_.push_back(std::make_shared<cv_wl_titlebar>(this));
    widget_geometries_.emplace_back(0, 0, 0, 0);

    viewer_ = std::make_shared<cv_wl_viewer>(this, flags);
    widget_geometries_.emplace_back(0, 0, 0, 0);
}

cv_wl_window::~cv_wl_window()
{
    if (frame_callback_)
        wl_callback_destroy(frame_callback_);
    xdg_surface_destroy(shell_surface_);
    wl_surface_destroy(surface_);
}

cv::Size cv_wl_window::get_size() const
{
    return size_;
}

std::string const& cv_wl_window::get_title() const
{
    return title_;
}

void cv_wl_window::set_title(std::string const& title)
{
    title_ = title;
    xdg_surface_set_title(shell_surface_, title_.c_str());
}

cv_wl_window_state const& cv_wl_window::state() const
{
    return state_;
}

cv_wl_buffer* cv_wl_window::next_buffer()
{
    cv_wl_buffer *buffer = nullptr;

    if (!buffers_.at(0).is_busy())
        buffer = &buffers_[0];
    else if (!buffers_.at(1).is_busy())
        buffer = &buffers_[1];

    return buffer;
}

void cv_wl_window::set_mouse_callback(CvMouseCallback on_mouse, void *param)
{
    viewer_->set_mouse_callback(on_mouse, param);
}

void cv_wl_window::set_minimized()
{
    xdg_surface_set_minimized(shell_surface_);
}

void cv_wl_window::set_maximized(bool maximize)
{
    if (!maximize)
        xdg_surface_unset_maximized(shell_surface_);
    else if (!(viewer_->get_flags() == cv::WINDOW_AUTOSIZE))
        xdg_surface_set_maximized(shell_surface_);
}

void cv_wl_window::show_image(cv::Mat const& image)
{
    viewer_->set_image(image);
    this->show();
}

void cv_wl_window::create_trackbar(std::string const& name, int *value, int count, CvTrackbarCallback2 on_change, void *userdata)
{
    auto exists = this->get_trackbar(name).lock();
    if (!exists) {
        auto trackbar =
            std::make_shared<cv_wl_trackbar>(
                this, name,value, count, on_change, userdata
            );
        widgets_.emplace_back(trackbar);
        widget_geometries_.emplace_back(0, 0, 0, 0);
    }
}

weak_ptr<cv_wl_trackbar> cv_wl_window::get_trackbar(std::string const& trackbar_name) const
{
    auto it = std::find_if(widgets_.begin(), widgets_.end(),
        [&trackbar_name](shared_ptr<cv_wl_widget> const& widget) {
            if (auto trackbar = std::dynamic_pointer_cast<cv_wl_trackbar>(widget))
                return trackbar->name() == trackbar_name;
            return false;
        });
    return it == widgets_.end() ? shared_ptr<cv_wl_trackbar>()
        : std::static_pointer_cast<cv_wl_trackbar>(*it);
}

static void calculate_damage(cv::Rect& surface_damage,
    cv::Rect const& widget_geometry, cv::Rect const& w_damage)
{
    if (w_damage.area() == 0)
        return;

    auto widget_damage = w_damage;
    widget_damage.x += widget_geometry.x;
    widget_damage.y += widget_geometry.y;

    if (surface_damage.area() == 0) {
        surface_damage = widget_damage;
    } else {
        auto damage = cv::Rect(0, 0, 0, 0);
        damage.x = std::min(surface_damage.x, widget_damage.x);
        damage.y = std::min(surface_damage.y, widget_damage.y);
        damage.width = std::max(surface_damage.x + surface_damage.width, widget_damage.x + widget_damage.width) - damage.x;
        damage.height = std::max(surface_damage.y + surface_damage.height, widget_damage.y + widget_damage.height) - damage.y;

        surface_damage = damage;
    }
}

std::tuple<cv::Size, std::vector<cv::Rect>>
cv_wl_window::manage_widget_geometry(cv::Size const& new_size)
{
    std::vector<cv::Rect> geometries;

    std::vector<int> min_widths, nat_widths;
    int min_width, nat_width, min_height, nat_height;

    auto store_preferred_width = [&](shared_ptr<cv_wl_widget> const& widget) {
        widget->get_preferred_width(min_width, nat_width);
        min_widths.push_back(min_width);
        nat_widths.push_back(nat_width);
    };

    store_preferred_width(viewer_);
    for (auto& widget : widgets_)
        store_preferred_width(widget);

    int final_width = 0;
    int total_height = 0;
    std::function<void (shared_ptr<cv_wl_widget> const&, int, bool)> calc_geometries;

    auto calc_autosize_geo = [&](shared_ptr<cv_wl_widget> const& widget, int width, bool) {
        widget->get_preferred_height_for_width(width, min_height, nat_height);
        geometries.emplace_back(0, total_height, width, nat_height);
        total_height += nat_height;
    };
    auto calc_normal_geo = [&](shared_ptr<cv_wl_widget> const& widget, int width, bool viewer) {
        widget->get_preferred_height_for_width(width, min_height, nat_height);
        int height = viewer ? (new_size.height - total_height) : nat_height;
        geometries.emplace_back(0, total_height, width, height);
        total_height += height;
    };

    if (viewer_->get_flags() == cv::WINDOW_AUTOSIZE) {
        final_width = nat_widths[0];
        calc_geometries = calc_autosize_geo;
    } else {
        int total_min_height = 0;
        int max_min_width = *std::max_element(min_widths.begin(), min_widths.end());
        auto calc_total_min_height = [&](shared_ptr<cv_wl_widget> const& widget) {
            widget->get_preferred_height_for_width(max_min_width, min_height, nat_height);
            total_min_height += min_height;
        };

        calc_total_min_height(viewer_);
        for (auto& widget : widgets_)
            calc_total_min_height(widget);

        auto min_size = cv::Size(max_min_width, total_min_height);
        if (new_size.width < min_size.width || new_size.height < min_size.height) {
            /* The new_size is smaller than the minimum size */
            return std::make_tuple(cv::Size(0, 0), geometries);
        } else {
            final_width = new_size.width;
            calc_geometries = calc_normal_geo;
        }
    }

    for (auto& widget : widgets_)
        calc_geometries(widget, final_width, false);
    calc_geometries(viewer_, final_width, true);

    return std::make_tuple(cv::Size(final_width, total_height), geometries);
}

void cv_wl_window::show(cv::Size const& size)
{
    auto *buffer = this->next_buffer();
    if (!next_frame_ready_ || !buffer) {
        if (size.area() == 0) {
            pending_.repaint_request = true;
        } else {
            pending_.size = size;
            pending_.resize_request = true;
        }
        return;
    }

    auto placement =
        this->manage_widget_geometry(size.area() == 0 ? size_ : size);
    auto new_size = std::get<0>(placement);
    auto const& geometries = std::get<1>(placement);
    if (new_size.area() == 0 || geometries.size() != (widgets_.size() + 1))
        return;

    bool buffer_size_changed = (buffer->size() != new_size);
    if (!buffer->is_allocated() || buffer_size_changed)
        buffer->create_shm(display_->shm(), new_size, WL_SHM_FORMAT_XRGB8888);

    auto surface_damage = cv::Rect(0, 0, 0, 0);
    auto draw_widget = [&](shared_ptr<cv_wl_widget> const& widget, cv::Rect const& rect) {
        auto widget_damage = widget->draw(
            buffer->data() + ((new_size.width * rect.y + rect.x) * 4),
            rect.size(),
            buffer_size_changed
        );
        calculate_damage(surface_damage, rect, widget_damage);
    };

    for (size_t i = 0; i < widgets_.size(); ++i)
        draw_widget(widgets_[i], geometries[i]);
    draw_widget(viewer_, geometries.back());

    this->commit_buffer(buffer, surface_damage);

    widget_geometries_ = std::move(geometries);
    size_ = new_size;
}

void cv_wl_window::commit_buffer(cv_wl_buffer *buffer, cv::Rect const& damage)
{
    if (!buffer)
        return;

    buffer->attach_to_surface(surface_, 0, 0);
    wl_surface_damage(surface_, damage.x, damage.y, damage.width, damage.height);

    if (frame_callback_)
        wl_callback_destroy(frame_callback_);
    frame_callback_ = wl_surface_frame(surface_);
    wl_callback_add_listener(frame_callback_, &frame_listener_, this);

    next_frame_ready_ = false;
    wl_surface_commit(surface_);
}

void cv_wl_window::handle_frame_callback(void *data, struct wl_callback *cb, uint32_t time)
{
    auto *window = reinterpret_cast<cv_wl_window *>(data);

    window->next_frame_ready_ = true;

    if (window->pending_.resize_request) {
        window->pending_.resize_request = false;
        window->pending_.repaint_request = false;
        window->show(window->pending_.size);
    } else if (window->pending_.repaint_request) {
        window->pending_.repaint_request = false;
        window->show();
    }
}

#define EDGE_AREA_MARGIN 7
static std::string get_cursor_name(int x, int y, cv::Size const& size, bool grab)
{
    std::string cursor;

    if (grab) {
        cursor = "grabbing";
    } else if (0 <= y && y <= EDGE_AREA_MARGIN) {
        cursor = "top_";
        if (0 <= x && x <= EDGE_AREA_MARGIN)
            cursor += "left_corner";
        else if (size.width - EDGE_AREA_MARGIN <= x && x <= size.width)
            cursor += "right_corner";
        else
            cursor += "side";
    } else if (size.height - EDGE_AREA_MARGIN <= y && y <= size.height) {
        cursor = "bottom_";
        if (0 <= x && x <= EDGE_AREA_MARGIN)
            cursor += "left_corner";
        else if (size.width - EDGE_AREA_MARGIN <= x && x <= size.width)
            cursor += "right_corner";
        else
            cursor += "side";
    } else if (0 <= x && x <= EDGE_AREA_MARGIN) {
        cursor = "left_side";
    } else if (size.width - EDGE_AREA_MARGIN <= x && x <= size.width) {
        cursor = "right_side";
    } else {
        cursor = "left_ptr";
    }

    return cursor;
}

static xdg_surface_resize_edge cursor_name_to_enum(std::string const& cursor)
{

    if (cursor == "top_left_corner") return XDG_SURFACE_RESIZE_EDGE_TOP_LEFT;
    else if (cursor == "top_right_corner") return XDG_SURFACE_RESIZE_EDGE_TOP_RIGHT;
    else if (cursor == "top_side") return XDG_SURFACE_RESIZE_EDGE_TOP;
    else if (cursor == "bottom_left_corner") return XDG_SURFACE_RESIZE_EDGE_BOTTOM_LEFT;
    else if (cursor == "bottom_right_corner") return XDG_SURFACE_RESIZE_EDGE_BOTTOM_RIGHT;
    else if (cursor == "bottom_side") return XDG_SURFACE_RESIZE_EDGE_BOTTOM;
    else if (cursor == "left_side") return XDG_SURFACE_RESIZE_EDGE_LEFT;
    else if (cursor == "right_side") return XDG_SURFACE_RESIZE_EDGE_RIGHT;
    else return XDG_SURFACE_RESIZE_EDGE_NONE;
}

void cv_wl_window::update_cursor(cv::Point const& p, bool grab)
{
    auto cursor_name = get_cursor_name(p.x, p.y, size_, grab);
    if (cursor_.current_name == cursor_name)
        return;

    cursor_.current_name = cursor_name;
    auto cursor = cursor_.theme.get_cursor(cursor_name);
    cursor.lock()->set_to_mouse(
        *display_->input().lock()->mouse().lock(),
        mouse_enter_serial_
    );
    cursor.lock()->commit();
}

void cv_wl_window::interactive_move()
{
    xdg_surface_move(
        shell_surface_,
        display_->input().lock()->seat(),
        mouse_button_serial_
    );
}

static int get_kb_modifiers_flag(weak_ptr<cv_wl_keyboard> kb)
{
    int flag = 0;
    auto modifiers = kb.lock()->get_modifiers();

    if (modifiers & cv_wl_keyboard::MOD_CONTROL_MASK)
        flag |= cv::EVENT_FLAG_CTRLKEY;
    if (modifiers & cv_wl_keyboard::MOD_ALT_MASK)
        flag |= cv::EVENT_FLAG_ALTKEY;
    if (modifiers & cv_wl_keyboard::MOD_SHIFT_MASK)
        flag |= cv::EVENT_FLAG_SHIFTKEY;

    return flag;
}

void cv_wl_window::deliver_mouse_event(int event, cv::Point const& p, int flag)
{
    flag |= get_kb_modifiers_flag(display_->input().lock()->keyboard());

    for (size_t i = 0; i < widgets_.size(); ++i) {
        auto const& rect = widget_geometries_[i];
        if (rect.contains(p))
            widgets_[i]->on_mouse(event, p - rect.tl(), flag);
    }

    auto const& rect = widget_geometries_.back();
    if (viewer_ && rect.contains(p))
        viewer_->on_mouse(event, p - rect.tl(), flag);
}

void cv_wl_window::mouse_enter(cv::Point const& p, uint32_t serial)
{
    on_mouse_.last = p;
    mouse_enter_serial_ = serial;

    this->update_cursor(p);
    this->deliver_mouse_event(cv::EVENT_MOUSEMOVE, p, 0);
}

void cv_wl_window::mouse_leave()
{
    on_mouse_.reset();
    cursor_.current_name.clear();
}

void cv_wl_window::mouse_motion(uint32_t time, cv::Point const& p)
{
    int flag = 0;
    on_mouse_.last = p;

    if (on_mouse_.drag) {
        switch (on_mouse_.button) {
        case cv_wl_mouse::LBUTTON:
            flag = cv::EVENT_FLAG_LBUTTON;
            break;
        case cv_wl_mouse::RBUTTON:
            flag = cv::EVENT_FLAG_RBUTTON;
            break;
        case cv_wl_mouse::MBUTTON:
            flag = cv::EVENT_FLAG_MBUTTON;
            break;
        default:
            break;
        }
    }

    bool grabbing =
        (cursor_.current_name == "grabbing" && (flag & cv::EVENT_FLAG_LBUTTON));
    this->update_cursor(p, grabbing);
    this->deliver_mouse_event(cv::EVENT_MOUSEMOVE, p, flag);
}

void cv_wl_window::mouse_button(uint32_t time, uint32_t button, wl_pointer_button_state state, uint32_t serial)
{
    int event = 0, flag = 0;

    mouse_button_serial_ = serial;

    /* Start a user-driven, interactive resize of the surface */
    if (WL_POINTER_BUTTON_STATE_PRESSED && !on_mouse_.drag &&
        button == cv_wl_mouse::LBUTTON && cursor_.current_name != "left_ptr" &&
         !(viewer_->get_flags() == cv::WINDOW_AUTOSIZE)) {
        xdg_surface_resize(
            shell_surface_,
            display_->input().lock()->seat(),
            serial,
            cursor_name_to_enum(cursor_.current_name)
        );
        return;
    }

    on_mouse_.button = static_cast<cv_wl_mouse::button>(button);
    on_mouse_.drag = (state == WL_POINTER_BUTTON_STATE_PRESSED);

    switch (button) {
    case cv_wl_mouse::LBUTTON:
        event = on_mouse_.drag ? cv::EVENT_LBUTTONDOWN : cv::EVENT_LBUTTONUP;
        flag = cv::EVENT_FLAG_LBUTTON;
        break;
    case cv_wl_mouse::RBUTTON:
        event = on_mouse_.drag ? cv::EVENT_RBUTTONDOWN : cv::EVENT_RBUTTONUP;
        flag = cv::EVENT_FLAG_RBUTTON;
        break;
    case cv_wl_mouse::MBUTTON:
        event = on_mouse_.drag ? cv::EVENT_MBUTTONDOWN : cv::EVENT_MBUTTONUP;
        flag = cv::EVENT_FLAG_MBUTTON;
        break;
    default:
        break;
    }

    this->update_cursor(on_mouse_.last);
    this->deliver_mouse_event(event, on_mouse_.last, flag);
}

void cv_wl_window::handle_surface_configure(
    void *data, struct xdg_surface *surface,
    int32_t width, int32_t height, struct wl_array *states, uint32_t serial)
{
    void *p;
    auto size = cv::Size(width, height);
    auto *window = reinterpret_cast<cv_wl_window *>(data);

    auto old_state = window->state_;
    window->state_.reset();

    xdg_surface_ack_configure(surface, serial);

    wl_array_for_each(p, states) {
        uint32_t state = *(reinterpret_cast<uint32_t *>(p));
        switch (state) {
        case XDG_SURFACE_STATE_MAXIMIZED:
            window->state_.maximized = true;
            if (!old_state.maximized) {
                window->state_.prev_size_ = window->size_;
                window->show(size);
            }
            break;
        case XDG_SURFACE_STATE_FULLSCREEN:
            window->state_.fullscreen = true;
            break;
        case XDG_SURFACE_STATE_RESIZING:
            window->state_.resizing = true;
            if (size.area() != 0)
                window->show(size);
            break;
        case XDG_SURFACE_STATE_ACTIVATED:
            window->state_.focused = true;
            break;
        default:
            /* Unknown state */
            break;
        }
    }

    /* When unmaximized, resize to the previous size */
    if (old_state.maximized && !window->state_.maximized)
        window->show(old_state.prev_size_);

#ifndef NDEBUG
    std::cerr << "[*] DEBUG: " << __func__
        << ": maximized=" << window->state_.maximized
        << " fullscreen=" << window->state_.fullscreen
        << " resizing=" << window->state_.resizing
        << " focused=" << window->state_.focused
        << " size=" << size << std::endl;
#endif
}

void cv_wl_window::handle_surface_close(void *data, struct xdg_surface *surface)
{
    //auto *window = reinterpret_cast<cv_wl_window *>(data);
}


/*
 * cv_wl_core implementation
 */
cv_wl_core::cv_wl_core()
{
}

cv_wl_core::~cv_wl_core()
{
    this->destroy_all_windows();
    display_.reset();
}

void cv_wl_core::init()
{
    display_ = std::make_shared<cv_wl_display>();
    if (!display_)
        CV_Error(StsNoMem, "Could not create display");
}

shared_ptr<cv_wl_display> cv_wl_core::display()
{
    return display_;
}

std::vector<std::string> cv_wl_core::get_window_names() const
{
    std::vector<std::string> names;
    for (auto&& e : windows_)
        names.emplace_back(e.first);
    return names;
}

shared_ptr<cv_wl_window> cv_wl_core::get_window(std::string const& name)
{
    return windows_.count(name) >= 1 ?
        windows_.at(name) : std::shared_ptr<cv_wl_window>();
}

void *cv_wl_core::get_window_handle(std::string const& name)
{
    auto window = get_window(name);
    return window ? get_window(name).get() : nullptr;
}

std::string const& cv_wl_core::get_window_name(void *handle)
{
    return handles_[handle];
}

bool cv_wl_core::create_window(std::string const& name, int flags)
{
    auto window = std::make_shared<cv_wl_window>(display_, name, flags);
    auto result = windows_.insert(std::make_pair(name, window));
    handles_[window.get()] = window->get_title();
    return result.second;
}

bool cv_wl_core::destroy_window(std::string const& name)
{
    return windows_.erase(name);
}

void cv_wl_core::destroy_all_windows()
{
    return windows_.clear();
}


/*                              */
/*  OpenCV highgui interfaces   */
/*                              */

/* Global wayland core object */
shared_ptr<cv_wl_core> cv_core;

CV_IMPL int cvInitSystem(int argc, char **argv)
{
    if (!cv_core) {
        cv_core = std::make_shared<cv_wl_core>();
        if (!cv_core)
            CV_ErrorNoReturn(StsNoMem, "Couldn't allocate enough memory");

        cv_core->init();
    }

    return 0;
}

CV_IMPL int cvStartWindowThread()
{
    cvInitSystem(0, NULL);

    return 0;
}

CV_IMPL int cvNamedWindow(const char *name, int flags)
{
    cvInitSystem(1, (char **)&name);

    return cv_core->create_window(name, flags);
}

CV_IMPL void cvDestroyWindow(const char* name)
{
    cvInitSystem(0, NULL);

    cv_core->destroy_window(name);
}

CV_IMPL void cvDestroyAllWindows()
{
    cvInitSystem(0, NULL);

    cv_core->destroy_all_windows();
}

CV_IMPL void* cvGetWindowHandle(const char* name)
{
    cvInitSystem(0, NULL);

    return cv_core->get_window_handle(name);
}

CV_IMPL const char* cvGetWindowName(void* window_handle)
{
    cvInitSystem(0, NULL);

    return cv_core->get_window_name(window_handle).c_str();
}

CV_IMPL void cvMoveWindow(const char* name, int x, int y)
{
    /*
     * We cannot move window surfaces not with the user actions in Wayland
     * Only a wayland compositor is allowed to do it
     * So this function is not implemented
     */
}

CV_IMPL void cvResizeWindow(const char* name, int width, int height)
{
    cvInitSystem(0, NULL);

    if (auto window = cv_core->get_window(name))
        window->show(cv::Size(width, height));
}

CV_IMPL int cvCreateTrackbar(const char* name_bar, const char* window_name, int* value, int count, CvTrackbarCallback on_change)
{
    return 0;
}

CV_IMPL int cvCreateTrackbar2(const char* trackbar_name, const char* window_name, int* val, int count, CvTrackbarCallback2 on_notify, void* userdata)
{
    cvInitSystem(0, NULL);

    if (auto window = cv_core->get_window(window_name))
        window->create_trackbar(trackbar_name, val, count, on_notify, userdata);

    return 0;
}

CV_IMPL int cvGetTrackbarPos(const char* trackbar_name, const char* window_name)
{
    cvInitSystem(0, NULL);

    if (auto window = cv_core->get_window(window_name)) {
        auto trackbar_ptr = window->get_trackbar(trackbar_name);
        if (auto trackbar = trackbar_ptr.lock())
            return trackbar->get_pos();
    }

    return -1;
}

CV_IMPL void cvSetTrackbarPos(const char* trackbar_name, const char* window_name, int pos)
{
    cvInitSystem(0, NULL);

    if (auto window = cv_core->get_window(window_name)) {
        auto trackbar_ptr = window->get_trackbar(trackbar_name);
        if (auto trackbar = trackbar_ptr.lock())
            trackbar->set_pos(pos);
    }
}

CV_IMPL void cvSetTrackbarMax(const char* trackbar_name, const char* window_name, int maxval)
{
    cvInitSystem(0, NULL);

    if (auto window = cv_core->get_window(window_name)) {
        auto trackbar_ptr = window->get_trackbar(trackbar_name);
        if (auto trackbar = trackbar_ptr.lock())
            trackbar->set_max(maxval);
    }
}

CV_IMPL void cvSetMouseCallback(const char* window_name, CvMouseCallback on_mouse, void* param)
{
    cvInitSystem(0, NULL);

    if (auto window = cv_core->get_window(window_name))
        window->set_mouse_callback(on_mouse, param);
}

CV_IMPL void cvShowImage(const char* name, const CvArr* arr)
{
    cvInitSystem(0, NULL);

    auto window = cv_core->get_window(name);
    if (!window) {
        cv_core->create_window(name, cv::WINDOW_AUTOSIZE);
        if (!(window = cv_core->get_window(name)))
            CV_Error_(StsNoMem, ("Failed to create window: %s", name));
    }

    cv::Mat mat = cv::cvarrToMat(arr, true);
    window->show_image(mat);
}

void cv::setWindowTitle(const String& winname, const String& title)
{
    cvInitSystem(0, NULL);

    if (auto window = cv_core->get_window(winname))
        window->set_title(title);
}

CV_IMPL int cvWaitKey(int delay)
{
    cvInitSystem(0, NULL);

    int key = -1;
    auto limit = ch::milliseconds(delay);

    while (true) {
        auto start = ch::steady_clock::now();

        std::pair<uint32_t, bool> events =
            cv_core->display()->run_once(
                limit.count() > 0 ? limit.count() : -1);

        if (events.first & EPOLLIN) {
            auto&& key_queue =
                cv_core->display()->input().lock()
                    ->keyboard().lock()->get_key_queue();
            if (!key_queue.empty()) {
                key = key_queue.back();
                break;
            }
        }

        // timeout
        auto end = ch::steady_clock::now();
        auto elapsed = ch::duration_cast<ch::milliseconds>(end - start);
        if (events.second || (limit.count() > 0 && elapsed >= limit))
            break;
        limit -= elapsed;
    }

    return key;
}

#ifdef HAVE_OPENGL
CV_IMPL void cvSetOpenGlDrawCallback(const char*, CvOpenGlDrawCallback, void*)
{
}

CV_IMPL void cvSetOpenGlContext(const char*)
{
}

CV_IMPL void cvUpdateWindow(const char*)
{
}
#endif // HAVE_OPENGL

#endif // HAVE_WAYLAND
#endif // _WIN32
