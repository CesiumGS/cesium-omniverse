#include "cesium/omniverse/CertificateAuthorityPaths.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace cesium::omniverse {

std::string getCAFile() {
    static std::filesystem::path result;

    if (!result.empty()) {
        return result.generic_string();
    }

#ifndef _WIN32
    // From https"//golang.org/src/crypto/x509/root_linux.go
    std::vector<std::filesystem::path> caFiles = {
        "/etc/ssl/certs/ca-certificates.crt",                // Debian/Ubuntu/Gentoo etc.
        "/etc/pki/tls/certs/ca-bundle.crt",                  // Fedora/RHEL 6
        "/etc/ssl/ca-bundle.pem",                            // OpenSUSE
        "/etc/pki/tls/cacert.pem",                           // OpenELEC
        "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem", // CentOS/RHEL 7
    };

    // SSL_CERT_FILE overrides defaults
    auto sslCertFileEnv = getenv("SSL_CERT_FILE");

    if (sslCertFileEnv) {
        caFiles = {std::filesystem::path(sslCertFileEnv)};
    }

    for (const auto& caFile : caFiles) {
        if (std::filesystem::exists(caFile)) {
            result = caFile;
            break;
        }
    }
#endif
    return result.generic_string();
}

std::string getCAPath() {
    static std::filesystem::path result;

    if (!result.empty()) {
        return result.generic_string();
    }

#ifndef _WIN32
    // From https"//golang.org/src/crypto/x509/root_unix.go
    std::vector<std::filesystem::path> caPaths = {
        "/etc/ssl/certs",               // SLES10/SLES11, https://golang.org/issue/12139
        "/system/etc/security/cacerts", // Android
        "/usr/local/share/certs",       // FreeBSD
        "/etc/pki/tls/certs",           // Fedora/RHEL
        "/etc/openssl/certs",           // NetBSD
    };

    // SSL_CERT_DIR overrides defaults
    auto sslCertDirEnv = getenv("SSL_CERT_DIR");

    if (sslCertDirEnv) {
        caPaths = {std::filesystem::path(sslCertDirEnv)};
    }

    for (const auto& caPath : caPaths) {
        if (std::filesystem::exists(caPath) && std::filesystem::is_directory(caPath)) {
            result = caPath;
            break;
        }
    }
#endif
    return result.generic_string();
}

} // namespace cesium::omniverse
