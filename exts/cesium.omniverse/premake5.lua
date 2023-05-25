local ext = get_current_extension_info()

project_ext (ext)

-- Link only those files and folders into the extension target directory
repo_build.prebuild_link { "bin", ext.target_dir.."/bin" }
repo_build.prebuild_link { "certs", ext.target_dir.."/certs" }
repo_build.prebuild_link { "cesium", ext.target_dir.."/cesium" }
repo_build.prebuild_link { "doc", ext.target_dir.."/doc" }
repo_build.prebuild_link { "images", ext.target_dir.."/images" }
