/*
 * Wayland Backend
 * TODO:
 *  Resizing
 *  Support WINDOW_NORMAL in cv_wl_window and cv_wl_viewer
 *  Support Cursor theme
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
#include <wayland-version.h>
#include "xdg-shell-client-protocol.h"
#include <xkbcommon/xkbcommon.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define BACKEND_NAME "OpenCV Wayland"

/*                              */
/*  OpenCV highgui internals    */
/*                              */
class cv_wl_display;
class cv_wl_input;
class cv_wl_mouse;
class cv_wl_keyboard;
class cv_wl_buffer;
class cv_wl_widget;
class cv_wl_viewer;
class cv_wl_trackbar;
class cv_wl_window;
class cv_wl_core;

using std::weak_ptr;
using std::shared_ptr;
using namespace cv::Error;
namespace ch = std::chrono;

extern shared_ptr<cv_wl_core> cv_core;

template<typename Container>
static void clear_container(Container& cont)
{
    Container empty_container;
    std::swap(cont, empty_container);
}

static void throw_system_error(std::string const& errmsg, int err)
{
    throw std::system_error(err, std::system_category(), errmsg);
}

static int xkb_keysym_to_ascii(xkb_keysym_t keysym)
{
    /* Remove most significant 8 bytes (0xff00) */
    return static_cast<uint8_t>(keysym);
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

    void init();
    static void handle_reg_global(void *data, struct wl_registry *reg, uint32_t name, const char *iface, uint32_t version);
    static void handle_reg_remove(void *data, struct wl_registry *wl_registry, uint32_t name);
    static void handle_shm_format(void *data, struct wl_shm *wl_shm, uint32_t format);
    static void handle_shell_ping(void *data, struct xdg_shell *shell, uint32_t serial);
};

class cv_wl_mouse {
public:
    enum button {
        NONE = 0,
        LBUTTON = 272,
        RBUTTON = 273,
        MBUTTON = 274,
    };

    cv_wl_mouse(struct wl_pointer *pointer);
    ~cv_wl_mouse();

private:
    struct wl_pointer *pointer_;
    struct wl_pointer_listener pointer_listener_{
        &handle_pointer_enter, &handle_pointer_leave,
        &handle_pointer_motion, &handle_pointer_button,
        &handle_pointer_axis
    };
    std::queue<cv_wl_window *> entered_window_;

    static void handle_pointer_enter(void *data, struct wl_pointer *pointer, uint32_t serial, struct wl_surface *surface, wl_fixed_t sx, wl_fixed_t sy);
    static void handle_pointer_leave(void *data, struct wl_pointer *pointer, uint32_t serial, struct wl_surface *surface);
    static void handle_pointer_motion(void *data, struct wl_pointer *pointer, uint32_t time, wl_fixed_t sx, wl_fixed_t sy);
    static void handle_pointer_button(void *data, struct wl_pointer *wl_pointer, uint32_t serial, uint32_t time, uint32_t button, uint32_t state);
    static void handle_pointer_axis(void *data, struct wl_pointer *wl_pointer, uint32_t time, uint32_t axis, wl_fixed_t value);
};

class cv_wl_keyboard {
public:
    cv_wl_keyboard(struct wl_keyboard *keyboard);
    ~cv_wl_keyboard();

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
    static int number_;
    std::string shm_path_;
    bool busy_ = false;
    cv::Size size_{0, 0};
    struct wl_buffer *buffer_ = nullptr;
    struct wl_buffer_listener buffer_listener_{
        &handle_buffer_release
    };
    void *shm_data_ = nullptr;

    static void handle_buffer_release(void *data, struct wl_buffer *buffer);
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

    virtual int get_preferred_width() const = 0;
    virtual int get_preferred_height_for_width(int width) const = 0;

    virtual void on_mouse(int event, int x, int y, int flag) {}

    /* Return: The area widget rendered, if not rendered at all, set as width=height=0 */
    virtual cv::Rect draw(void *data, cv::Size const&, bool force) = 0;

protected:
    cv::Size last_size_{0, 0};
    cv_wl_window *window_;
};

class cv_wl_viewer : public cv_wl_widget {
public:
    enum {
        MOUSE_CALLBACK_MIN_INTERVAL_MILLISEC = 15
    };

    cv_wl_viewer(cv_wl_window *, int flags);

    void set_image(cv::Mat const& img);
    void set_mouse_callback(CvMouseCallback callback, void *param);

    int get_preferred_width() const override;
    int get_preferred_height_for_width(int width) const override;
    void on_mouse(int event, int x, int y, int flag) override;
    cv::Rect draw(void *data, cv::Size const&, bool force) override;

private:
    int flags_;
    cv::Mat image_;
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

    int get_preferred_width() const override;
    int get_preferred_height_for_width(int width) const override;
    void on_mouse(int event, int x, int y, int flag) override;
    cv::Rect draw(void *data, cv::Size const& size, bool force) override;

private:
    std::string name_;
    int *value_, count_;
    cv::Size size_;

    struct {
        CvTrackbarCallback2 callback;
        void *data;
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
    int last_x = 0, last_y = 0;
    cv_wl_mouse::button button = cv_wl_mouse::button::NONE;
};

class cv_wl_window {
public:
    cv_wl_window(shared_ptr<cv_wl_display> const& display, std::string const& name, int flags);
    ~cv_wl_window();

    cv::Size get_size() const;
    std::string const& name() const;

    void print_debug_info() const;

    void show_image(cv::Mat const& image);

    void create_trackbar(std::string const& name, int *value, int count, CvTrackbarCallback2 on_change, void *userdata);
    weak_ptr<cv_wl_trackbar> get_trackbar(std::string const&) const;

    void mouse_enter(int x, int y);
    void mouse_leave();
    void mouse_motion(uint32_t time, int x, int y);
    void mouse_button(uint32_t time, uint32_t button, wl_pointer_button_state state);

    void set_mouse_callback(CvMouseCallback on_mouse, void *param);

    void show();

private:
    cv::Size size_{0, 0};
    std::string const name_;

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

    struct {
        bool repaint_request = false;  /* we need to redraw as soon as possible (some states are changed) */
    } pending_;

    shared_ptr<cv_wl_viewer> viewer_;
    cv::Point viewer_point_{0, 0};
    std::vector<shared_ptr<cv_wl_widget>> widgets_;
    std::vector<cv::Point> widgets_points_;

    cv_wl_mouse_callback on_mouse_;

    cv_wl_buffer* next_buffer();
    void commit_buffer(cv_wl_buffer *buffer, cv::Rect const&);
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
    :   display_{wl_display_connect(nullptr)}
{
    init();
}

cv_wl_display::cv_wl_display(std::string const& disp)

    :   display_{wl_display_connect(disp.c_str())}
{
    init();
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

void cv_wl_display::init()
{
    if (!display_)
        throw_system_error("Could not connect to display", errno);

    registry_ = wl_display_get_registry(display_);
    wl_registry_add_listener(registry_, &reg_listener_, this);
    wl_display_roundtrip(display_);
    if (!compositor_ || !shm_ || !shell_ || !input_)
        throw std::runtime_error("Compositor doesn't have required interfaces");

    wl_display_roundtrip(display_);
    if (!(formats_ & (1 << WL_SHM_FORMAT_XRGB8888)))
        throw std::runtime_error("WL_SHM_FORMAT_XRGB32 not available");

    poller_.add(
        wl_display_get_fd(display_),
        EPOLLIN | EPOLLOUT | EPOLLERR | EPOLLHUP
     );
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

void cv_wl_mouse::handle_pointer_enter(void *data, struct wl_pointer *pointer,
    uint32_t serial, struct wl_surface *surface, wl_fixed_t sx, wl_fixed_t sy)
{
    int x = wl_fixed_to_int(sx);
    int y = wl_fixed_to_int(sy);
    auto *mouse = reinterpret_cast<cv_wl_mouse *>(data);
    auto *window = reinterpret_cast<cv_wl_window *>(wl_surface_get_user_data(surface));

    mouse->entered_window_.push(window);
    window->mouse_enter(x, y);
}

void cv_wl_mouse::handle_pointer_leave(void *data,
    struct wl_pointer *pointer, uint32_t serial, struct wl_surface *surface)
{
    auto *mouse = reinterpret_cast<cv_wl_mouse *>(data);
    auto *window = reinterpret_cast<cv_wl_window *>(wl_surface_get_user_data(surface));

    window->mouse_leave();
    mouse->entered_window_.pop();
}

void cv_wl_mouse::handle_pointer_motion(void *data,
    struct wl_pointer *pointer, uint32_t time, wl_fixed_t sx, wl_fixed_t sy)
{
    int x = wl_fixed_to_int(sx);
    int y = wl_fixed_to_int(sy);
    auto *mouse = reinterpret_cast<cv_wl_mouse *>(data);
    auto *window = mouse->entered_window_.front();

    window->mouse_motion(time, x, y);
}

void cv_wl_mouse::handle_pointer_button(void *data, struct wl_pointer *wl_pointer,
    uint32_t serial, uint32_t time, uint32_t button, uint32_t state)
{
    auto *mouse = reinterpret_cast<cv_wl_mouse *>(data);
    auto *window = mouse->entered_window_.front();

    window->mouse_button(time, button, static_cast<wl_pointer_button_state>(state));
}

void cv_wl_mouse::handle_pointer_axis(void *data, struct wl_pointer *wl_pointer,
    uint32_t time, uint32_t axis, wl_fixed_t value)
{
}


/*
 * cv_wl_keyboard implementation
 */
cv_wl_keyboard::cv_wl_keyboard(struct wl_keyboard *keyboard)
    :   keyboard_(keyboard)
{
    xkb_.ctx = xkb_context_new(XKB_CONTEXT_NO_FLAGS);
    if (!xkb_.ctx)
        throw std::runtime_error("Failed to create xkb context");
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
    if (!seat_)
        throw std::runtime_error("Invalid seat detected when initializing");
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
    if (!mouse_)
        throw std::runtime_error("seat: mouse not available");
    return mouse_;
}

weak_ptr<cv_wl_keyboard> cv_wl_input::keyboard()
{
    if (!keyboard_)
        throw std::runtime_error("seat: keyboard not available");
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

int cv_wl_viewer::get_preferred_width() const
{
    return image_.size().width;
}

int cv_wl_viewer::get_preferred_height_for_width(int width) const
{
    /* Keep the aspect ratio */
    return image_.size().area() == 0 ? 0
        : (double)width * ((double)image_.size().height / (double)image_.size().width);
}

void cv_wl_viewer::on_mouse(int event, int x, int y, int flag)
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

            callback_(event, x, y, flag, param_);
        }
    }
}

cv::Rect cv_wl_viewer::draw(void *data, cv::Size const& size, bool force)
{
    if ((!force && !image_changed_) || image_.size().area() == 0 || size.area() == 0)
        return cv::Rect(0, 0, 0, 0);

    if (flags_ & cv::WINDOW_AUTOSIZE) {
        assert(image_.size() == size);
        write_mat_to_xrgb8888(image_, data);
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
    :   cv_wl_widget(window), name_(name), value_(value), count_(count)
{
    on_change_.callback = on_change;
    on_change_.data = data;
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

int cv_wl_trackbar::get_preferred_width() const
{
    return 320;  /* minimum and natural width */
}

int cv_wl_trackbar::get_preferred_height_for_width(int width) const
{
    return 40;  /* minimum and natural width */
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

    if (slider_moved_ || force) {
        size_ = last_size_ = size;

        if (size_ == data_.size())
            data_ = CV_RGB(0xde, 0xde, 0xde);
        else
            data_ = cv::Mat(size_, CV_8UC3, CV_RGB(0xde, 0xde, 0xde));

        this->prepare_to_draw();
        cv::putText(
            data_,
            (name_ + ": " + std::to_string(slider_.value)).c_str(),
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

void cv_wl_trackbar::on_mouse(int event, int x, int y, int flag)
{
    switch (event) {
    case cv::EVENT_LBUTTONDOWN:
        slider_.drag = true;
        break;
    case cv::EVENT_MOUSEMOVE:
        if (!(flag & cv::EVENT_FLAG_LBUTTON))
            break;
    case cv::EVENT_LBUTTONUP:
        if (slider_.drag && bar_.left.x <= x && x <= bar_.right.x) {
            slider_.value = (double)(x - bar_.left.x) / bar_.length() * count_;
            slider_moved_ = true;
            window_->show();
            slider_.drag = (event != cv::EVENT_LBUTTONUP);
        }
    }
}


/*
 * cv_wl_buffer implementation
 */
int cv_wl_buffer::number_ = 0;

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

    if (shm_data_ && shm_data_ != MAP_FAILED) {
        munmap(shm_data_, size_.area() * 4);
        shm_data_ = nullptr;
    }

    size_.width = size_.height = 0;
    shm_unlink(shm_path_.c_str());
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

    shm_path_ = "/opencv_shm-" + std::to_string(number_++);
    int fd = shm_open(shm_path_.c_str(), O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
    if (fd < 0)
        throw_system_error("failed to create a shared memory", errno);

    if (ftruncate(fd, buffer_size) < 0) {
        int errno_ = errno;
        close(fd);
        throw_system_error("failed to truncate a shm buffer", errno_);
    }

    shm_data_ = mmap(NULL, buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (shm_data_ == MAP_FAILED) {
        int errno_ = errno;
        close(fd);
        this->destroy();
        throw_system_error("failed to map shm", errno_);
    }

    struct wl_shm_pool *pool = wl_shm_create_pool(shm, fd, buffer_size);
    buffer_ = wl_shm_pool_create_buffer(pool, 0, size_.width, size_.height, stride, format);
    wl_buffer_add_listener(buffer_, &buffer_listener_, this);
    wl_shm_pool_destroy(pool);
    close(fd);
}

void cv_wl_buffer::attach_to_surface(struct wl_surface *surface, int32_t x, int32_t y)
{
    wl_surface_attach(surface, buffer_, x, y);
    this->busy();
}

void cv_wl_buffer::handle_buffer_release(void *data, struct wl_buffer *buffer)
{
    auto *cvbuf = reinterpret_cast<cv_wl_buffer *>(data);

    cvbuf->busy(false);
}


/*
 * cv_wl_window implementation
 */
cv_wl_window::cv_wl_window(shared_ptr<cv_wl_display> const& display, std::string const& name, int flags)
    : name_(name), display_(display), surface_(display->get_surface())
{
    shell_surface_ = display->get_shell_surface(surface_);
    if (!shell_surface_)
        throw std::runtime_error("Failed to get xdg_surface");

    xdg_surface_add_listener(shell_surface_, &shsurf_listener, this);
    xdg_surface_set_title(shell_surface_, name_.c_str());

    wl_surface_set_user_data(surface_, this);

    viewer_ = std::make_shared<cv_wl_viewer>(this, flags);
    viewer_point_ = cv::Point(0, 0);
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

std::string const& cv_wl_window::name() const
{
    return name_;
}

cv_wl_buffer* cv_wl_window::next_buffer()
{
    cv_wl_buffer *buffer;

    if (!buffers_.at(0).is_busy())
        buffer = &buffers_[0];
    else if (!buffers_.at(1).is_busy())
        buffer = &buffers_[1];
    else
        return nullptr;

    return buffer;
}

void cv_wl_window::set_mouse_callback(CvMouseCallback on_mouse, void *param)
{
    viewer_->set_mouse_callback(on_mouse, param);
}

void cv_wl_window::show_image(cv::Mat const& image)
{
    viewer_->set_image(image);
}

void cv_wl_window::create_trackbar(std::string const& name, int *value, int count, CvTrackbarCallback2 on_change, void *userdata)
{
    auto trackbar =
        std::make_shared<cv_wl_trackbar>(
            this, name,value, count, on_change, userdata
        );
    widgets_.emplace_back(trackbar);
    widgets_points_.emplace_back(0, 0);
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

static void calculate_damage(cv::Rect& surface_damage, cv::Rect const& widget_damage, int curr_height = 0)
{
    if (widget_damage.area() != 0) {
        surface_damage.x = std::min(surface_damage.x, widget_damage.x);
        surface_damage.y = std::min(surface_damage.y, curr_height + widget_damage.y);
        surface_damage.width = std::max(surface_damage.x + surface_damage.width, widget_damage.x + widget_damage.width);
        surface_damage.height = std::max(surface_damage.height, curr_height + widget_damage.y + widget_damage.height - surface_damage.y);
    }
}

void cv_wl_window::show()
{
    auto *buffer = this->next_buffer();
#if !defined(NDEBUG)
    std::cerr << "[*] DEBUG: buffer0@" << std::hex << &buffers_[0] << ".busy=" << buffers_[0].is_busy()
        << " buffer1@" << &buffers_[1] << ".busy=" << buffers_[1].is_busy()
        << " pending_.repaint_request=" << pending_.repaint_request
        << " next_frame_ready=" << next_frame_ready_
        << " buffer_available=" << (buffer ? true : false)
        << " buffer=" << buffer << std::dec
        << std::endl;
#endif
    if (!next_frame_ready_ || !buffer) {
        pending_.repaint_request = true;
        return;
    }

    int total_height = 0;
    int max_width = viewer_->get_preferred_width();
    std::vector<int> height_list;

    /* Ask each widget how long width do they need */
    /* And fit with the longest width (mostly the viewer) */
    for (auto& widget : widgets_)
        max_width = std::max(max_width, widget->get_preferred_width());

    /* Ask each widget how long height do they need for the given width */
    /* The total height becomes the actual height of the window */
    for (auto& widget : widgets_) {
        height_list.push_back(widget->get_preferred_height_for_width(max_width));
        total_height += height_list.back();
    }
    height_list.push_back(viewer_->get_preferred_height_for_width(max_width));
    total_height += height_list.back();

    /* The actual size of a surface */
    size_ = cv::Size(max_width, total_height);

    bool buffer_size_changed = (buffer->size() != size_);
    if (!buffer->is_allocated() || buffer_size_changed)
        buffer->create_shm(display_->shm(), size_, WL_SHM_FORMAT_XRGB8888);

    int curr_height = 0;
    auto surface_damage = cv::Rect(cv::Point(size_), cv::Size(0, 0));
    for (size_t i = 0; i < widgets_.size(); ++i) {
        auto widget_damage = widgets_[i]->draw(
            buffer->data() + (max_width * curr_height * 4),
            cv::Size(max_width, height_list[i]),
            buffer_size_changed
        );

        calculate_damage(surface_damage, widget_damage, curr_height);

        widgets_points_[i] = cv::Point(0, curr_height);
        curr_height += height_list[i];
    }

    auto viewer_damage = viewer_->draw(
        buffer->data() + (max_width * curr_height * 4),
        cv::Size(max_width, height_list.back()),
        buffer_size_changed
    );
    calculate_damage(surface_damage, viewer_damage, curr_height);
    viewer_point_ = cv::Point(0, curr_height);

    this->commit_buffer(buffer, surface_damage);
}

void cv_wl_window::print_debug_info() const
{
#if !defined(NDEBUG)
    std::cerr << "[*] DEBUG: buffer0@" << std::hex << &buffers_[0] << ".busy=" << buffers_[0].is_busy()
        << " buffer1@" << &buffers_[1] << ".busy=" << buffers_[1].is_busy()
        << " pending_.repaint_request=" << pending_.repaint_request
        << " next_frame_ready=" << next_frame_ready_
        << " buffer_size=" << size_
        << std::endl;
#endif
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

    if (window->pending_.repaint_request) {
        window->pending_.repaint_request = false;
        window->show();
    }
}

void cv_wl_window::mouse_enter(int x, int y)
{
    on_mouse_.last_x = x;
    on_mouse_.last_y = y;

    for (size_t i = 0; i < widgets_.size(); ++i) {
        auto size = widgets_[i]->get_last_size();
        auto&& p = widgets_points_[i];
        if (p.y <= y && y <= p.y + size.height)
            widgets_[i]->on_mouse(cv::EVENT_MOUSEMOVE, x, y - p.y, 0);
    }

    if (viewer_ && viewer_point_.y <= y)
        viewer_->on_mouse(cv::EVENT_MOUSEMOVE, x, y - viewer_point_.y, 0);
}

void cv_wl_window::mouse_leave()
{
}

void cv_wl_window::mouse_motion(uint32_t time, int x, int y)
{
    int flag = 0;
    on_mouse_.last_x = x;
    on_mouse_.last_y = y;

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

    for (size_t i = 0; i < widgets_.size(); ++i) {
        auto size = widgets_[i]->get_last_size();
        auto&& p = widgets_points_[i];
        if (p.y <= y && y <= p.y + size.height)
            widgets_[i]->on_mouse(cv::EVENT_MOUSEMOVE, x, y - p.y, flag);
    }

    if (viewer_ && viewer_point_.y <= y)
        viewer_->on_mouse(cv::EVENT_MOUSEMOVE, x, y - viewer_point_.y, flag);
}

void cv_wl_window::mouse_button(uint32_t time, uint32_t button, wl_pointer_button_state state)
{
    int event = 0, flag = 0;
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

    for (size_t i = 0; i < widgets_.size(); ++i) {
        auto size = widgets_[i]->get_last_size();
        auto const& p = widgets_points_[i];
        if (p.y <= on_mouse_.last_y && on_mouse_.last_y <= p.y + size.height)
            widgets_[i]->on_mouse(event, on_mouse_.last_x, on_mouse_.last_y - p.y, flag);
    }

    if (viewer_ && viewer_point_.y <= on_mouse_.last_y)
        viewer_->on_mouse(event, on_mouse_.last_x, on_mouse_.last_y - viewer_point_.y, flag);
}

void cv_wl_window::handle_surface_configure(
    void *data, struct xdg_surface *surface,
    int32_t width, int32_t height, struct wl_array *states, uint32_t serial)
{
    //auto *window = reinterpret_cast<cv_wl_window *>(data);
    xdg_surface_ack_configure(surface, serial);
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
        throw std::runtime_error("Could not create display");
    display_->roundtrip();
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
    return windows_.at(name);
}

void *cv_wl_core::get_window_handle(std::string const& name)
{
    return get_window(name).get();
}

std::string const& cv_wl_core::get_window_name(void *handle)
{
    return handles_[handle];
}

bool cv_wl_core::create_window(std::string const& name, int flags)
{
    auto window = std::make_shared<cv_wl_window>(display_, name, flags);
    auto result = windows_.insert(std::make_pair(name, window));
    handles_[window.get()] = window->name();
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
try {
    if (!cv_core) {
        cv_core = std::make_shared<cv_wl_core>();
        if (!cv_core)
            CV_ErrorNoReturn(StsNoMem, "Couldn't allocate enough memory");

        cv_core->init();
    }

    return 0;
} catch (std::exception& e) {
    CV_ErrorNoReturn(StsInternal, e.what());
}

CV_IMPL int cvStartWindowThread()
try {
    cvInitSystem(0, NULL);

    return 0;
} catch (std::exception& e) {
    CV_ErrorNoReturn(StsInternal, e.what());
}

CV_IMPL int cvNamedWindow(const char *name, int flags)
try {
    cvInitSystem(1, (char **)&name);

    return cv_core->create_window(name, flags);
} catch (std::exception& e) {
    CV_ErrorNoReturn(StsInternal, e.what());
}

CV_IMPL void cvDestroyWindow(const char* name)
try {
    cvInitSystem(0, NULL);

    cv_core->destroy_window(name);
} catch (std::exception& e) {
    CV_ErrorNoReturn(StsInternal, e.what());
}

CV_IMPL void cvDestroyAllWindows()
try {
    cvInitSystem(0, NULL);

    cv_core->destroy_all_windows();
} catch (std::exception& e) {
    CV_ErrorNoReturn(StsInternal, e.what());
}

CV_IMPL void* cvGetWindowHandle(const char* name)
try {
    cvInitSystem(0, NULL);

    return cv_core->get_window_handle(name);
} catch (std::exception& e) {
    CV_ErrorNoReturn(StsInternal, e.what());
}

CV_IMPL const char* cvGetWindowName(void* window_handle)
try {
    cvInitSystem(0, NULL);

    return cv_core->get_window_name(window_handle).c_str();
} catch (std::exception& e) {
    CV_ErrorNoReturn(StsInternal, e.what());
}

CV_IMPL void cvMoveWindow(const char* name, int x, int y)
try {
    /*
     * We cannot move window surfaces not with the user actions in Wayland
     * Only a wayland compositor is allowed to do it
     * So this function is not implemented
     */
} catch (std::exception& e) {
    CV_ErrorNoReturn(StsInternal, e.what());
}

CV_IMPL void cvResizeWindow(const char* name, int width, int height)
try {
    cvInitSystem(0, NULL);
} catch (std::exception& e) {
    CV_ErrorNoReturn(StsInternal, e.what());
}

CV_IMPL int cvCreateTrackbar(const char* name_bar, const char* window_name, int* value, int count, CvTrackbarCallback on_change)
try {
    //auto window = cv_core->get_window(window_name);

    //window->create_trackbar(name_bar, value, count, on_change, nullptr);
    return 0;
} catch (std::exception& e) {
    CV_ErrorNoReturn(StsInternal, e.what());
}

CV_IMPL int cvCreateTrackbar2(const char* trackbar_name, const char* window_name, int* val, int count, CvTrackbarCallback2 on_notify, void* userdata)
try {
    cvInitSystem(0, NULL);

    auto window = cv_core->get_window(window_name);
    window->create_trackbar(trackbar_name, val, count, on_notify, userdata);

    return 0;
} catch (std::exception& e) {
    CV_ErrorNoReturn(StsInternal, e.what());
}

CV_IMPL int cvGetTrackbarPos(const char* trackbar_name, const char* window_name)
try {
    cvInitSystem(0, NULL);

    auto window = cv_core->get_window(window_name);

    auto trackbar_ptr = window->get_trackbar(trackbar_name);
    if (auto trackbar = trackbar_ptr.lock())
        return trackbar->get_pos();

    return -1;
} catch (std::exception& e) {
    CV_ErrorNoReturn(StsInternal, e.what());
}

CV_IMPL void cvSetTrackbarPos(const char* trackbar_name, const char* window_name, int pos)
try {
    cvInitSystem(0, NULL);

    auto window = cv_core->get_window(window_name);

    auto trackbar_ptr = window->get_trackbar(trackbar_name);
    if (auto trackbar = trackbar_ptr.lock())
        trackbar->set_pos(pos);
} catch (std::exception& e) {
    CV_ErrorNoReturn(StsInternal, e.what());
}

CV_IMPL void cvSetTrackbarMax(const char* trackbar_name, const char* window_name, int maxval)
try {
    cvInitSystem(0, NULL);

    auto window = cv_core->get_window(window_name);

    auto trackbar_ptr = window->get_trackbar(trackbar_name);
    if (auto trackbar = trackbar_ptr.lock())
        trackbar->set_max(maxval);
} catch (std::exception& e) {
    CV_ErrorNoReturn(StsInternal, e.what());
}

CV_IMPL void cvSetMouseCallback(const char* window_name, CvMouseCallback on_mouse, void* param)
try {
    cvInitSystem(0, NULL);

    auto window = cv_core->get_window(window_name);

    window->set_mouse_callback(on_mouse, param);
} catch (std::exception& e) {
    CV_ErrorNoReturn(StsInternal, e.what());
}

CV_IMPL void cvShowImage(const char* name, const CvArr* arr)
try {
    cvInitSystem(0, NULL);

    shared_ptr<cv_wl_window> window;
    try {
        window = cv_core->get_window(name);
    } catch (std::out_of_range& e) {
        cv_core->create_window(name, cv::WINDOW_AUTOSIZE);
        window = cv_core->get_window(name);
    }

    cv::Mat mat = cv::cvarrToMat(arr, true);
    window->show_image(mat);
    window->show();
} catch (std::exception& e) {
    CV_ErrorNoReturn(StsInternal, e.what());
}

CV_IMPL int cvWaitKey(int delay)
try {
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
} catch (std::exception& e) {
    CV_ErrorNoReturn(StsInternal, e.what());
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
