//#define __OPENCV_BUILD
//#define CV_IMPL
#include "precomp.hpp"

#ifndef _WIN32
#if defined (HAVE_WAYLAND)

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <functional>
#include <memory>
#include <system_error>

#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <linux/input.h>

#include <wayland-client.h>
#include <wayland-client-protocol.h>
#include <wayland-cursor.h>
#include <wayland-util.h>
#include <wayland-version.h>
#include "xdg-shell-client-protocol.h"

#include <xkbcommon/xkbcommon.h>

#define BACKEND_NAME "[DEBUG] OpenCV Wayland"

#define DEBUG_PRINT_LOCATION_INFO \
    std::cerr << BACKEND_NAME << ": " << __func__ << ": " << __FILE__ << ":" << __LINE__ << " passed" << std::endl

/*                              */
/*  OpenCV highgui internals    */
/*                              */
class cv_wl_display;
class cv_wl_input;
class cv_wl_keyboard;
class cv_wl_buffer;
class cv_wl_window;
class cv_wl_core;

using std::shared_ptr;
extern shared_ptr<cv_wl_core> g_core;

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
 *	determine the xkb keycode, clients must add 8 to the key event keycode
 */
static xkb_keycode_t xkb_keycode_from_raw_keycode(int raw_keycode)
{
    return raw_keycode + 8;
}

static void draw_argb8888(void *d, uint8_t a, uint8_t r, uint8_t g, uint8_t b)
{
    *((uint32_t *)d) = ((a << 24) | (r << 16) | (g << 8) | b);
}

static void write_mat_to_xrgb8888(CvMat const *mat, void *data)
{
    std::cerr << BACKEND_NAME << ": " << __func__ << ": writing image: " << mat->cols << "x" << mat->rows << " step=" << mat->step << std::endl;
    int x, y;
    for (y = 0; y < mat->rows; y++) {
        for (x = 0; x < mat->cols; x++) {
            uint8_t p[3];
            p[0] = mat->data.ptr[mat->step * y + x * 3];
            p[1] = mat->data.ptr[mat->step * y + x * 3 + 1];
            p[2] = mat->data.ptr[mat->step * y + x * 3 + 2];
            draw_argb8888((char *)data + (y * mat->cols + x) * 4, 0x00, p[2], p[1], p[0]);
        }
    }
    std::cerr << BACKEND_NAME << ": " << __func__ << ": finished writing" << std::endl;
}

class cv_wl_display {
public:
    cv_wl_display();
    cv_wl_display(std::string const& disp);
    ~cv_wl_display();

    int dispatch();
    int dispatch_pending();
    int flush();
    int roundtrip();
    struct wl_shm *shm();
    struct wl_seat *seat();
    shared_ptr<cv_wl_input> input();
    uint32_t formats() const;
    struct wl_surface *get_surface();
    struct xdg_surface *get_shell_surface(struct wl_surface *surface);

private:
    struct wl_display *display_;
    struct wl_registry *registry_;
    struct wl_registry_listener reglistener_{&handle_reg_global, &handle_reg_remove};
    struct wl_compositor *compositor_ = nullptr;
    struct wl_shm *shm_ = nullptr;
    struct wl_shm_listener shmlistener_{&handle_shm_format};
    struct xdg_shell *shell_ = nullptr;
    struct xdg_shell_listener shell_listener_{&handle_shell_ping};
    struct wl_seat *seat_ = nullptr;
    shared_ptr<cv_wl_input> input_;
    uint32_t formats_ = 0;

    void init();
    static void handle_reg_global(void *data, struct wl_registry *reg, uint32_t name, const char *iface, uint32_t version);
    static void handle_reg_remove(void *data, struct wl_registry *wl_registry, uint32_t name);
    static void handle_shm_format(void *data, struct wl_shm *wl_shm, uint32_t format);
    static void handle_shell_ping(void *data, struct xdg_shell *shell, uint32_t serial);
};

class cv_wl_keyboard {
public:
    cv_wl_keyboard(struct wl_keyboard *keyboard);
    ~cv_wl_keyboard();

    int wait_key(int delay);

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

    shared_ptr<cv_wl_keyboard> keyboard();

private:
    struct wl_seat *seat_;
    struct wl_seat_listener seat_listener_{
        &handle_seat_capabilities, &handle_seat_name
    };
    shared_ptr<cv_wl_keyboard> keyboard_;

    static void handle_seat_capabilities(void *data, struct wl_seat *wl_seat, uint32_t capabilities);
    static void handle_seat_name(void *data, struct wl_seat *wl_seat, const char *name);
};

class cv_wl_buffer {
public:
    struct wl_buffer *buffer = nullptr;
    void *shm_data = nullptr;

    ~cv_wl_buffer();
    void destroy();
    void busy(bool busy = true);
    bool is_busy() const;
    int width() const;
    int height() const;
    int create_shm(struct wl_shm *shm, int width, int height, uint32_t format);

private:
    static int number_;
    bool busy_ = false;
    int width_ = 0, height_ = 0;
    struct wl_buffer_listener buffer_listener_ = {&handle_buffer_release};
    std::string shm_path_;

    /* 'busy' means 'buffer' is being used by a compositor */
    static void handle_buffer_release(void *data, struct wl_buffer *buffer);
};

class cv_wl_window {
public:
    enum {
        default_width = 320,
        default_height = 240
    };

    cv_wl_window(shared_ptr<cv_wl_display> display, std::string const& name, int flags);
    cv_wl_window(shared_ptr<cv_wl_display> display, std::string const& name, int width, int height, int flags);
    ~cv_wl_window();

    std::string const& name() const;
    std::pair<int, int> get_size() const;
    void show_image(CvMat const *mat);

private:
    int const flags_;
    std::string const name_;
    int width_, height_;

    shared_ptr<cv_wl_display> display_;
    struct wl_surface *surface_;
    struct xdg_surface *shell_surface_;
    struct xdg_surface_listener surface_listener_{
        &handle_surface_configure, &handle_surface_close
    };
    /* double-buffered */
    cv_wl_buffer buffers_[2];

    cv_wl_buffer& next_buffer();

    static void handle_surface_configure(void *, struct xdg_surface *, int32_t, int32_t, struct wl_array *, uint32_t);
    static void handle_surface_close(void *data, struct xdg_surface *xdg_surface);
};

class cv_wl_core {
public:
    cv_wl_core();
    ~cv_wl_core();

    void init();

    shared_ptr<cv_wl_display> display();
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
    std::cerr << BACKEND_NAME << ": " << __func__ << ": dtor called" << std::endl;
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

struct wl_shm *cv_wl_display::shm()
{
    return shm_;
}

struct wl_seat *cv_wl_display::seat()
{
    return seat_;
}

shared_ptr<cv_wl_input> cv_wl_display::input()
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
        throw_system_error("Could not connect to display: ", errno);

    registry_ = wl_display_get_registry(display_);
    wl_registry_add_listener(registry_, &reglistener_, this);
    wl_display_roundtrip(display_);
    if (!compositor_ || !shm_ || !shell_ || !seat_ || !input_)
        throw std::runtime_error("Compositor doesn't have required interfaces");

    wl_display_roundtrip(display_);
    if (!(formats_ & (1 << WL_SHM_FORMAT_XRGB8888)))
        throw std::runtime_error("WL_SHM_FORMAT_XRGB32 not available");
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
        wl_shm_add_listener(display->shm_, &display->shmlistener_, display);
    } else if (interface == "xdg_shell") {
        display->shell_ = (struct xdg_shell *)
            wl_registry_bind(reg, name, &xdg_shell_interface, version);
        xdg_shell_use_unstable_version(display->shell_, XDG_SHELL_VERSION_CURRENT);
        xdg_shell_add_listener(display->shell_, &display->shell_listener_, display);
    } else if (interface == "wl_seat") {
        display->seat_ = (struct wl_seat *)
            wl_registry_bind(reg, name, &wl_seat_interface, version);
        display->input_ = std::make_shared<cv_wl_input>(display->seat_);
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
 * cv_wl_keyboard implementation
 */
cv_wl_keyboard::cv_wl_keyboard(struct wl_keyboard *keyboard)
    :   keyboard_(keyboard)
{
    wl_keyboard_add_listener(keyboard_, &keyboard_listener_, this);
    xkb_.ctx = xkb_context_new(XKB_CONTEXT_NO_FLAGS);
    if (!xkb_.ctx)
        throw std::runtime_error("Failed to create xkb context");
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
    std::cerr << BACKEND_NAME << ": " << __func__ << ": dtor called" << std::endl;
}

int cv_wl_keyboard::wait_key(int delay)
{
    int key = -1;
    while (g_core->display()->dispatch() != -1) {
        if (!key_queue_.empty()) {
            key = key_queue_.back();
            break;
        }
    }
    return key;
}

void cv_wl_keyboard::handle_kb_keymap(void *data, struct wl_keyboard *kb, uint32_t format, int fd, uint32_t size)
{
    auto *keyboard = reinterpret_cast<cv_wl_keyboard *>(data);

    if (format != WL_KEYBOARD_KEYMAP_FORMAT_XKB_V1) {
        close(fd);
        return;
    }

    char *map_str = (char *)mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (map_str == MAP_FAILED) {
        close(fd);
        return;
    }

    keyboard->xkb_.keymap = xkb_keymap_new_from_string(
        keyboard->xkb_.ctx, map_str, XKB_KEYMAP_FORMAT_TEXT_V1, XKB_KEYMAP_COMPILE_NO_FLAGS);
    munmap(map_str, size);
    close(fd);
    if (!keyboard->xkb_.keymap) {
        std::cerr << "failed to compile keymap" << std::endl;
        return;
    }

    keyboard->xkb_.state = xkb_state_new(keyboard->xkb_.keymap);
    if (!keyboard->xkb_.state) {
        std::cerr << "failed to create XKB state" << std::endl;
        xkb_keymap_unref(keyboard->xkb_.keymap);
        return;
    }

    keyboard->xkb_.control_mask =
        1 << xkb_keymap_mod_get_index(keyboard->xkb_.keymap, "Control");
    keyboard->xkb_.alt_mask =
        1 << xkb_keymap_mod_get_index(keyboard->xkb_.keymap, "Mod1");
    keyboard->xkb_.shift_mask =
        1 << xkb_keymap_mod_get_index(keyboard->xkb_.keymap, "Shift");
}

void cv_wl_keyboard::handle_kb_enter(void *data, struct wl_keyboard *keyboard, uint32_t serial, struct wl_surface *surface, struct wl_array *keys)
{
}

void cv_wl_keyboard::handle_kb_leave(void *data, struct wl_keyboard *keyboard, uint32_t serial, struct wl_surface *surface)
{
}

void cv_wl_keyboard::handle_kb_key(void *data, struct wl_keyboard *keyboard, uint32_t serial, uint32_t time, uint32_t key, uint32_t state)
{
    xkb_keycode_t keycode = xkb_keycode_from_raw_keycode(key);
    auto *kb = reinterpret_cast<cv_wl_keyboard *>(data);

    std::cerr << "Key is " << key << " state is " << state << std::endl;
    if (state == WL_KEYBOARD_KEY_STATE_RELEASED) {
        xkb_keysym_t keysym = xkb_state_key_get_one_sym(kb->xkb_.state, keycode);
        kb->key_queue_.push(xkb_keysym_to_ascii(keysym));
        std::cerr << __func__ << ": ASCII=" << kb->key_queue_.back() << "queued" << std::endl;
    }

}

void cv_wl_keyboard::handle_kb_modifiers(void *data, struct wl_keyboard *keyboard,
                        uint32_t serial, uint32_t mods_depressed,
                        uint32_t mods_latched, uint32_t mods_locked,
                        uint32_t group)
{
    fprintf(stderr, "Modifiers depressed %d, latched %d, locked %d, group %d\n",
        mods_depressed, mods_latched, mods_locked, group);
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
    keyboard_.reset();
    wl_seat_destroy(seat_);
    std::cerr << BACKEND_NAME << ": " << __func__ << ": dtor called" << std::endl;
}

shared_ptr<cv_wl_keyboard> cv_wl_input::keyboard()
{
    return keyboard_;
}

void cv_wl_input::handle_seat_capabilities(void *data, struct wl_seat *wl_seat, uint32_t caps)
{
    auto *input = reinterpret_cast<cv_wl_input *>(data);

    if (caps & WL_SEAT_CAPABILITY_KEYBOARD) {
        std::cerr << BACKEND_NAME << ": seat cap: keyboard available" << std::endl;
        struct wl_keyboard *keyboard = wl_seat_get_keyboard(input->seat_);
        input->keyboard_ = std::make_shared<cv_wl_keyboard>(keyboard);
    }
}

void cv_wl_input::handle_seat_name(void *data, struct wl_seat *wl_seat, const char *name)
{
    std::cerr << BACKEND_NAME << ": seat name: " << name << std::endl;
}


/*
 * cv_wl_buffer implementation
 */
int cv_wl_buffer::number_ = 0;

cv_wl_buffer::~cv_wl_buffer()
{
    this->destroy();
    std::cerr << BACKEND_NAME << ": " << __func__ << ": dtor called" << std::endl;
}

void cv_wl_buffer::destroy()
{
    if (buffer) {
        wl_buffer_destroy(buffer);
        buffer = nullptr;
        width_ = 0;
        height_ = 0;
    }
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

int cv_wl_buffer::width() const
{
    return width_;
}

int cv_wl_buffer::height() const
{
    return height_;
}

int cv_wl_buffer::create_shm(struct wl_shm *shm, int width, int height, uint32_t format)
{
    int stride = width * 4;
    int size = stride * height;
    struct wl_shm_pool *pool;

    this->destroy();
    this->width_ = width;
    this->height_ = height;

    shm_path_ = "/opencv_wl_buffer-" + std::to_string(number_++);
    int fd = shm_open(shm_path_.c_str(), O_RDWR | O_CREAT, 0700);
    if (fd < 0)
        throw_system_error("creating a buffer file failed: ", errno);

    if (ftruncate(fd, size) < 0)
        throw_system_error("failed to truncate a shm buffer", errno);

    shm_data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (shm_data == MAP_FAILED) {
        close(fd);
        throw_system_error("mmap failed: ", errno);
    }

    pool = wl_shm_create_pool(shm, fd, size);
    buffer =
        wl_shm_pool_create_buffer(pool, 0, width, height, stride, format);
    wl_buffer_add_listener(buffer, &buffer_listener_, this);
    wl_shm_pool_destroy(pool);
    close(fd);
    return 0;
}

void cv_wl_buffer::handle_buffer_release(void *data, struct wl_buffer *buffer)
{
    auto *mybuf = reinterpret_cast<cv_wl_buffer *>(data);
    mybuf->busy(false);
}


/*
 * cv_wl_window implementation
 */
cv_wl_window::cv_wl_window(shared_ptr<cv_wl_display> display, std::string const& name, int flags)
    :   cv_wl_window(display, name, default_width, default_height, flags)
{
}

cv_wl_window::cv_wl_window(shared_ptr<cv_wl_display> display,
    std::string const& name, int width, int height, int flags)
    :   flags_(flags), name_(name), width_(width), height_(height),
        display_(display), surface_(display->get_surface())
{
    shell_surface_ = display->get_shell_surface(surface_);
    xdg_surface_add_listener(shell_surface_, &surface_listener_, this);
    xdg_surface_set_title(shell_surface_, name_.c_str());

    wl_surface_set_user_data(surface_, this);
}

cv_wl_window::~cv_wl_window()
{
    xdg_surface_destroy(shell_surface_);
    wl_surface_destroy(surface_);
    std::cerr << BACKEND_NAME << ": " << __func__ << ": dtor called" << std::endl;
}

std::string const& cv_wl_window::name() const
{
    return name_;
}

std::pair<int, int> cv_wl_window::get_size() const
{
    return std::make_pair(width_, height_);
}

void cv_wl_window::show_image(CvMat const *mat)
{
    width_ = mat->cols;
    height_ = mat->rows;

    cv_wl_buffer& buffer = this->next_buffer();

    write_mat_to_xrgb8888(mat, buffer.shm_data);

    wl_surface_attach(surface_, buffer.buffer, 0, 0);
    wl_surface_damage(surface_, 0, 0, width_, height_);
    wl_surface_commit(surface_);
    buffer.busy();
}

cv_wl_buffer& cv_wl_window::next_buffer()
{
    cv_wl_buffer *buffer;

    if (!buffers_[0].is_busy())
        buffer = &buffers_[0];
    else if (!buffers_[1].is_busy())
        buffer = &buffers_[1];
    else
        throw std::runtime_error(
            "Both buffers are busy. Maybe a bug of a compositor?");

    if (!buffer->buffer || buffer->width() != width_ || buffer->height() != height_) {
        int ret = buffer->create_shm(display_->shm(),
            width_, height_, WL_SHM_FORMAT_XRGB8888);
        if (ret < 0)
            throw std::runtime_error("cannot create shm buffer");

        /* paint the padding */
        memset(buffer->shm_data, 0xff, width_ * height_ * 4);
    }
    return *buffer;
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
    std::cerr << BACKEND_NAME << ": " << __func__ << ": dtor called" << std::endl;
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
shared_ptr<cv_wl_core> g_core;

CV_IMPL int cvInitSystem(int argc, char **argv)
{
    if (!g_core) try {
        std::cerr << BACKEND_NAME << ": Initializing backend" << std::endl;
        g_core = std::make_shared<cv_wl_core>();
        g_core->init();
    } catch (...) {
        /* We just need to report an error */
    }
    return g_core ? 0 : -1;
}

CV_IMPL int cvStartWindowThread()
{
    std::cerr << BACKEND_NAME << ": " << __func__ << std::endl;
    return 0;
}

CV_IMPL int cvNamedWindow(const char *name, int flags)
{
    std::cerr << BACKEND_NAME << ": " << __func__ << ": " << name << std::endl;
    if (cvInitSystem(1, (char **)&name))
        return -1;

    return g_core->create_window(name, flags);
}

CV_IMPL void cvDestroyWindow(const char* name)
{
    std::cerr << BACKEND_NAME << ": " << __func__ << ": " << name << std::endl;
    g_core->destroy_window(name);
}

CV_IMPL void cvDestroyAllWindows()
{
    std::cerr << BACKEND_NAME << ": " << __func__ << std::endl;
    g_core->destroy_all_windows();
}

CV_IMPL void* cvGetWindowHandle(const char* name)
{
    std::cerr << BACKEND_NAME << ": " << __func__ << ": " << name << std::endl;
    return g_core->get_window_handle(name);
}

CV_IMPL const char* cvGetWindowName(void* window_handle)
{
    return g_core->get_window_name(window_handle).c_str();
}

CV_IMPL void cvMoveWindow(const char* name, int x, int y)
{
    /*
     * We cannot move window surfaces in Wayland
     * Only a wayland compositor is allowed to do it
     * So this function is not implemented
     */
}

CV_IMPL void cvResizeWindow(const char* name, int width, int height)
{
    /*
     * We cannot resize window surfaces in Wayland
     * Only a wayland compositor is allowed to do it
     * So this function is not implemented
     */
}

CV_IMPL int cvCreateTrackbar(const char* name_bar, const char* window_name, int* value, int count, CvTrackbarCallback on_change)
{
    return 0;
}

CV_IMPL int cvCreateTrackbar2(const char* name_bar, const char* window_name, int* val, int count, CvTrackbarCallback2 on_notify, void* userdata)
{
    return 0;
}

CV_IMPL int cvGetTrackbarPos(const char* name_bar, const char* window_name)
{
    return 0;
}

CV_IMPL void cvSetTrackbarPos(const char* name_bar, const char* window_name, int pos)
{
}

CV_IMPL int cvCreateButton(const char* button_name, CvButtonCallback on_change, void* userdata, int button_type, int initial_button_state)
{
    return 0;
}

CV_IMPL void cvSetMouseCallback(const char* window_name, CvMouseCallback on_mouse, void* param)
{
}

CV_IMPL void cvShowImage(const char* name, const CvArr* arr)
{
    std::cerr << BACKEND_NAME << ": " << __func__ << ": " << name << std::endl;
    CvMat stub;
    CvMat *mat = cvGetMat(arr, &stub);
    auto window = g_core->get_window(name);
    window->show_image(mat);
}

CV_IMPL int cvWaitKey(int delay)
{
    std::cerr << BACKEND_NAME << ": " << __func__ << ": delay=" << delay << std::endl;
    return g_core->display()->input()->keyboard()->wait_key(0);
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
