#define __OPENCV_BUILD

#define CvFont void 
//#define CV_IMPL 
//#include "precomp.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

#ifndef _WIN32

#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <functional>
#include <memory>

#include <wayland-client.h>
#include <wayland-client-protocol.h>
#include <wayland-cursor.h>
#include <wayland-util.h>
#include <wayland-version.h>


/*                              */
/*                              */
/*  OpenCV highgui internals    */
/*                              */

using std::shared_ptr;

class cv_wl_display {
public:
    cv_wl_display()
        :   display_{wl_display_connect(nullptr)}
    {
        init();
    }

    cv_wl_display(std::string const& disp)
        :   display_{wl_display_connect(disp.c_str())}
    {
        init();
    }

    struct wl_surface *get_surface()
    {
        return wl_compositor_create_surface(compositor_);
    }

    struct wl_shell_surface *get_shell_surface(struct wl_surface *surface)
    {
        return wl_shell_get_shell_surface(shell_, surface);
    }

private:
    struct wl_display *display_;
    struct wl_registry *registry_;
    struct wl_registry_listener reglistener_{&handle_reg_global, nullptr};
    struct wl_compositor *compositor_;
    struct wl_shm *shm_;
    struct wl_shell *shell_;
    uint32_t formats = WL_SHM_FORMAT_XRGB8888;

    void init()
    {
        registry_ = wl_display_get_registry(display_);
        wl_registry_add_listener(registry_, &reglistener_, this);
    }

    static void handle_reg_global(void *data, struct wl_registry *reg, uint32_t name, const char *iface, uint32_t version)
    {
        auto *core = reinterpret_cast<cv_wl_display *>(data);
        std::string const interface = iface;
        if (interface == "wl_compositor") {
            core->compositor_ = (struct wl_compositor *)
                wl_registry_bind(reg, name, &wl_compositor_interface, version);
        } else if (interface == "wl_shm") {
            core->shm_ = (struct wl_shm *)
                wl_registry_bind(reg, name, &wl_shm_interface, version);
        } else if (interface == "wl_shell") {
            core->shell_ = (struct wl_shell *)
                wl_registry_bind(reg, name, &wl_shell_interface, version);
        }
    }
};

class cv_wl_buffer {
public:
    struct wl_buffer *buffer;
    void *shm_data;
    int busy;
};

class cv_wl_window {
public:
    cv_wl_window(shared_ptr<cv_wl_display> display, std::string const& name, int flags)
        :   flags_(flags), name_(name),
            display_(display), surface_(display->get_surface())
    {
        shell_surface_ = display->get_shell_surface(surface_);
        wl_shell_surface_add_listener(shell_surface_, &sslistener_, this);
    }

    std::string const& name()
    {
        return name_;
    }

private:
    int const flags_;
    std::string const name_;
    shared_ptr<cv_wl_display> display_;

    int height_ = 320, width_ = 240;
    struct wl_surface *surface_;
    struct wl_shell_surface *shell_surface_;
    struct wl_shell_surface_listener sslistener_{&handle_surface_ping};

    /* double-buffered */
    cv_wl_buffer buffer[2];
    cv_wl_buffer *prev_buffer;

    /* callback for next redrawing */
    struct wl_callback *callback;

    static void handle_surface_ping(void *data, struct wl_shell_surface *surface, uint32_t serial)
    {
        wl_shell_surface_pong(surface, serial);
    }
};

class cv_wl_core {
public:
    cv_wl_core()
        :   display_(std::make_shared<cv_wl_display>())
    {
        if (!display_)
            throw std::runtime_error("Could not create display_");
    }

    shared_ptr<cv_wl_window> get_window(std::string const& name)
    {
        return windows_.at(name);
    }

    void *get_window_handle(std::string const& name)
    {
        return get_window(name).get();
    }

    std::string const& get_window_name(void *handle)
    {
        return handles_[handle];
    }

    bool create_window(std::string const& name, int flags)
    {
        auto window = std::make_shared<cv_wl_window>(display_, name, flags);
        auto result = windows_.insert(std::make_pair(name, window));
        handles_[window.get()] = window->name();
        return result.second;
    }

    bool destroy_window(std::string const& name)
    {
        return windows_.erase(name);
    }

    void destroy_all_windows()
    {
        return windows_.clear();
    }

private:
    shared_ptr<cv_wl_display> display_;
    std::map<std::string, shared_ptr<cv_wl_window>> windows_;
    std::map<void *, std::string> handles_;
};

shared_ptr<cv_wl_core> g_core;  // Global wayland core object


/*                              */
/*                              */
/*  OpenCV highgui interfaces   */
/*                              */

CV_IMPL int cvInitSystem(int argc, char **argv)
{
    if (!g_core) try {
        g_core = std::make_shared<cv_wl_core>();
    } catch (std::exception& e) {
        /* We just need to report an error */
    }
    return g_core ? 0 : -1;
}

CV_IMPL int cvStartWindowThread()
{
    return 0;
}

CV_IMPL int cvNamedWindow(const char *name, int flags)
{
    if (cvInitSystem(1, (char**)&name))
        return -1;

    return g_core->create_window(name, flags);
}

CV_IMPL void cvDestroyWindow(const char* name)
{
    g_core->destroy_window(name);
}

CV_IMPL void cvDestroyAllWindows()
{
    g_core->destroy_all_windows();
}

CV_IMPL void* cvGetWindowHandle(const char* name)
{
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
}

CV_IMPL void cvUpdateWindow(const char* window_name)
{
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
}

CV_IMPL int cvWaitKey(int delay)
{
    return 0;
}


#endif // _WIN32
