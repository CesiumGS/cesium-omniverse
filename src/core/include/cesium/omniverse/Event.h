#pragma once

#include <algorithm>
#include <functional>
#include <type_traits>
#include <vector>

namespace Cesium {
template <class TYPE> using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<TYPE>>;

template <class... ARGS> class event_handler;

template <class... ARGS> class event final {
  public:
    event() = default;

    event(const event&) = delete;

    event(event&&) noexcept;

    event& operator=(const event&) = delete;

    event& operator=(event&&) noexcept;

    ~event() noexcept;

    [[nodiscard]] std::size_t num_of_handlers() const noexcept;

    void add_handler(event_handler<ARGS...>& handler);

    void remove_handler(event_handler<ARGS...>& handler);

    template <class... INVOKE_ARGS> void invoke(INVOKE_ARGS&&...);

  private:
    bool _is_invoking{false};
    std::vector<event_handler<ARGS...>*> _handlers;
};

template <class... ARGS> class event_handler final {
  public:
    event_handler() = default;

    template <
        class FUNCTION,
        typename FUNCTION_ENABLE = std::enable_if_t<!std::is_same_v<remove_cvref_t<FUNCTION>, event_handler>>>
    explicit event_handler(FUNCTION&& func);

    event_handler(const event_handler&) = delete;

    event_handler(event_handler&&) noexcept;

    event_handler& operator=(const event_handler&) = delete;

    event_handler& operator=(event_handler&&) noexcept;

    ~event_handler() noexcept;

    void connect(event<ARGS...>& event);

    void disconnect();

    [[nodiscard]] bool is_connected() const noexcept;

    [[nodiscard]] const event<ARGS...>* connected_event() const noexcept;

    [[nodiscard]] event<ARGS...>* connected_event() noexcept;

    template <class FUNCTION> void set_callback(FUNCTION&& func);

  private:
    event<ARGS...>* _event{nullptr};
    std::function<void(ARGS...)> _callback{};

    friend class event<ARGS...>;
};

/*****************************************************
 *
 *                  Inline definitions
 *
 *****************************************************/
template <class... ARGS>
event<ARGS...>::event(event&& rhs) noexcept
    : _is_invoking{rhs._is_invoking}
    , _handlers{std::move(rhs._handlers)} {
    for (auto handler : _handlers) {
        handler->_event = this;
    }
}

template <class... ARGS> event<ARGS...>& event<ARGS...>::operator=(event&& rhs) noexcept {
    using std::swap;
    swap(_is_invoking, rhs._is_invoking);
    swap(_handlers, rhs._handlers);

    for (auto handler : _handlers) {
        handler->_event = this;
    }

    for (auto handler : rhs._handlers) {
        handler->_event = &rhs;
    }

    return *this;
}

template <class... ARGS> event<ARGS...>::~event() noexcept {
    for (auto handler : _handlers) {
        handler->_event = nullptr;
    }
}

template <class... ARGS> size_t event<ARGS...>::num_of_handlers() const noexcept {
    return _handlers.size();
}

template <class... ARGS> void event<ARGS...>::add_handler(event_handler<ARGS...>& handler) {
    if (handler._event != this) {
        handler.disconnect();
    }

    if (handler._event == this) {
        return;
    }

    handler._event = this;
    _handlers.emplace_back(&handler);
}

template <class... ARGS> void event<ARGS...>::remove_handler(event_handler<ARGS...>& handler) {
    if (handler._event != this) {
        return;
    }

    handler._event = nullptr;

    if (!_is_invoking) {
        auto it = std::remove(_handlers.begin(), _handlers.end(), &handler);
        _handlers.erase(it, _handlers.end());
    } else {
        // mark for deletion to be cleaned up after invoke() is finished.
        // We can't delete it right away since it messes with the order of the handlers
        // during the invoke
        auto it = std::find(_handlers.begin(), _handlers.end(), &handler);
        *it = nullptr;
    }
}

template <class... ARGS> template <class... INVOKE_ARGS> void event<ARGS...>::invoke(INVOKE_ARGS&&... args) {
    _is_invoking = true;

    for (std::size_t i = 0; i < _handlers.size(); ++i) {
        // There is a good chance that more handlers are connected when the below callback is called.
        // But it should be fine since we copy the pointer of each handler here
        auto handler = _handlers[i];

        // Handler can be marked for deletion (by using nullptr) when another handler is invoked, so
        // we have to check it here before invoking the callback of the current handler
        if (handler && handler->_callback) {
            handler->_callback(std::forward<INVOKE_ARGS>(args)...);
        }
    }

    // remove any handlers that are marked for deletion during this invoke
    auto it = std::remove(_handlers.begin(), _handlers.end(), nullptr);
    _handlers.erase(it, _handlers.end());

    _is_invoking = false;
}

template <class... ARGS>
template <class FUNCTION, typename FUNCTION_ENABLE>
event_handler<ARGS...>::event_handler(FUNCTION&& func)
    : _callback{std::forward<FUNCTION>(func)} {}

template <class... ARGS>
event_handler<ARGS...>::event_handler(event_handler&& rhs) noexcept
    : _callback{std::move(rhs._callback)} {
    if (rhs._event) {
        connect(*rhs._event);
    }

    rhs.disconnect();
}

template <class... ARGS> event_handler<ARGS...>& event_handler<ARGS...>::operator=(event_handler&& rhs) noexcept {
    if (&rhs != this) {
        disconnect();

        _callback = std::move(rhs._callback);
        if (rhs._event) {
            connect(*rhs._event);
        }

        rhs.disconnect();
    }

    return *this;
}

template <class... ARGS> event_handler<ARGS...>::~event_handler() noexcept {
    disconnect();
}

template <class... ARGS> void event_handler<ARGS...>::connect(event<ARGS...>& event) {
    event.add_handler(*this);
}

template <class... ARGS> void event_handler<ARGS...>::disconnect() {
    if (_event) {
        _event->remove_handler(*this);
    }
}

template <class... ARGS> bool event_handler<ARGS...>::is_connected() const noexcept {
    return _event != nullptr;
}

template <class... ARGS> const event<ARGS...>* event_handler<ARGS...>::connected_event() const noexcept {
    return _event;
}

template <class... ARGS> event<ARGS...>* event_handler<ARGS...>::connected_event() noexcept {
    return _event;
}

template <class... ARGS> template <class FUNCTION> void event_handler<ARGS...>::set_callback(FUNCTION&& func) {
    _callback = std::forward<FUNCTION>(func);
}
} // namespace Cesium
