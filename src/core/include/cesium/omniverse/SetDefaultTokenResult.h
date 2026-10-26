#pragma once

#include <string>

namespace cesium::omniverse {

enum SetDefaultTokenResultCode {
    NOT_SET_IN_SESSION = -1,
    OK = 0,
    NOT_CONNECTED_TO_ION = 1,
    CREATE_FAILED = 2,
};

namespace SetDefaultTokenResultMessages {
inline constexpr std::string_view NOT_SET_IN_SESSION_MESSAGE = "Default token has not been set this session.";
inline constexpr std::string_view OK_MESSAGE = "OK";
inline constexpr std::string_view NOT_CONNECTED_TO_ION_MESSAGE = "Not connected to ion.";
inline constexpr std::string_view CREATE_FAILED_MESSAGE_BASE = "Create failed: {} ({})";
} // namespace SetDefaultTokenResultMessages

/**
 * Stores information about the last action to set the default token. A code and a relevant user
 * friendly message are stored.
 */
struct SetDefaultTokenResult {
    int code{SetDefaultTokenResultCode::NOT_SET_IN_SESSION};
    std::string message{SetDefaultTokenResultMessages::NOT_SET_IN_SESSION_MESSAGE};
};
} // namespace cesium::omniverse
