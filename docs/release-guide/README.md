# Releasing a new version of Cesium for Omniverse

This is the process we follow when releasing a new version of Cesium for Omniverse on GitHub.

1. Make sure the latest commit in `main` is passing CI.
2. Update the project `VERSION` in [CMakeLists.txt](../../CMakeLists.txt).
3. Update the extension `version` in [extension.toml](../../exts/cesium.omniverse/config/extension.toml). This should be the same version as above.
4. Update [`CHANGES.md`](../../exts/cesium.omniverse/doc/CHANGES.md).
5. Commit and push the changes to `main`: `git commit -am "0.0.0 release"`, `git push origin main`
6. Wait for CI to pass
7. In the AWS management console (old AWS account), go to the bucket [`cesium-travis-builds/cesium-omniverse/main`](https://s3.console.aws.amazon.com/s3/buckets/cesium-travis-builds?region=us-east-1&prefix=cesium-omniverse/main/&showversions=false), find the appropriate date and commit hash to download the CentOS and Windows builds zip files (e.g. `CesiumForOmniverse-Linux-vX.X.X.zip` and `CesiumForOmniverse-Windows-vX.X.X.zip` )
8. Verify that the Linux package loads in Omniverse Create (see instructions below).
9. Verify that the Windows package loads in Omniverse Create (see instructions below).
10. Tag the release, e.g., `git tag -a v0.0.0 -m "0.0.0 release"`.
11. Push the tag to github: `git push origin v0.0.0`.
12.  Create a new release on GitHub: https://github.com/CesiumGS/cesium-omniverse/releases/new.
  * Chose the new tag.
  * Copy the changelog into the description. Follow the format used in previous releases.
  * Upload the Linux and Windows packages.
13. Proceed to [Releasing a new version of Cesium for Omniverse Samples](#releasing-a-new-version-of-cesium-for-omniverse-samples).

# Releasing a new version of Cesium for Omniverse Samples

1. Create a new access token using the CesiumJS ion account.
    * The name of the token should match "Cesium for Omniverse Samples vX.X.X - Delete on April 1st, 2023" where the version is the same as the Cesium for Omniverse release and the expiry date is two months later than present.
    * The scope of the token should be "assets:read" for all assets.
2. Replace the `cesium:projectDefaultIonAccessToken` property in each `.usda` file with the new access token.
3. Verify that all the USD files load in Cesium for Omniverse.
4. Update `CHANGES.md`.
5. Commit and push the changes to `main`.
6. Tag the release, e.g., `git tag -a v0.0.0 -m "0.0.0 release"`.
7. Push the tag to github: `git push origin v0.0.0`.
8. Create a new release on GitHub: https://github.com/CesiumGS/cesium-omniverse-samples/releases/new.
    * Choose the new tag.
    * Copy the changelog into the description. Follow the format used in previous releases.

# Verify Package

After the package is built, verify that the extension loads in Omniverse Create:

* Open Omniverse Create
* Remove the existing Cesium for Omniverse extension from the list of search paths
* Close Omniverse Create
* Unzip the package to `$USERHOME$/Documents/Kit/Shared/exts`
* Open Omniverse Create
* Enable Cesium for Omniverse and check autoload
* Restart Omniverse Create
* Verify that there aren't any console errors
* Verify that you can load Cesium World Terrain and OSM buildings
* Delete the extensions from `$USERHOME$/Documents/Kit/Shared/exts`
