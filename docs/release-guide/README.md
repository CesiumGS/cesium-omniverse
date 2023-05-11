# Releasing a new version of Cesium for Omniverse

This is the process we follow when releasing a new version of Cesium for Omniverse on GitHub.

1. Make sure the latest commit in `main` is passing CI.
2. Download the latest build from S3. In the AWS management console (old AWS account), go to the bucket [`cesium-travis-builds/cesium-omniverse/main`](https://s3.console.aws.amazon.com/s3/buckets/cesium-travis-builds?region=us-east-1&prefix=cesium-omniverse/main/&showversions=false), find the appropriate date and commit hash to download the CentOS and Windows zip files (e.g. `CesiumForOmniverse-Linux-xxxxxxx.zip` and `CesiumForOmniverse-Windows-xxxxxxx.zip`)
3. Verify that the Linux package loads in Omniverse Create (see instructions below).
4. Verify that the Windows package loads in Omniverse Create (see instructions below).
5. Update the project `VERSION` in [CMakeLists.txt](../../CMakeLists.txt).
6. Update the extension `version` in [extension.toml](../../exts/cesium.omniverse/config/extension.toml). This should be the same version as above.
7. Update [`CHANGES.md`](CHANGES.md).
8. Commit the changes, e.g. `git commit -am "0.0.0 release"`.
9. Push the commit, e.g. `git push origin main`.
10. Tag the release, e.g. `git tag -a v0.0.0 -m "0.0.0 release"`.
11. Push the tag, e.g. `git push origin v0.0.0`.
12. Wait for CI to pass.
13. Download the latest build from S3. In the AWS management console (old AWS account), go to the bucket [`cesium-travis-builds/cesium-omniverse/main`](https://s3.console.aws.amazon.com/s3/buckets/cesium-travis-builds?region=us-east-1&prefix=cesium-omniverse/main/&showversions=false), find the appropriate date and commit hash to download the CentOS and Windows zip files (e.g. `CesiumForOmniverse-Linux-xxxxxxx.zip` and `CesiumForOmniverse-Windows-xxxxxxx.zip` )
14. Rename both zip files to use the version number instead of the commit hash, e.g. rename `CesiumForOmniverse-Linux-xxxxxxx.zip` to `CesiumForOmniverse-Linux-v0.0.0.zip` and `CesiumForOmniverse-Windows-xxxxxxx.zip` to `CesiumForOmniverse-Windows-v0.0.0.zip`.
15. Create a new release on GitHub: https://github.com/CesiumGS/cesium-omniverse/releases/new.
    * Chose the new tag.
    * Copy the changelog into the description. Follow the format used in previous releases.
    * Upload the Linux and Windows release zip files.
16. Proceed to [Releasing a new version of Cesium for Omniverse Samples](#releasing-a-new-version-of-cesium-for-omniverse-samples).

# Releasing a new version of Cesium for Omniverse Samples

1. Create a new access token using the CesiumJS ion account.
    * The name of the token should match "Cesium for Omniverse Samples vX.X.X - Delete on April 1st, 2023" where the version is the same as the Cesium for Omniverse release and the expiry date is two months later than present.
    * The scope of the token should be "assets:read" for all assets.
2. Replace the `cesium:projectDefaultIonAccessToken` property in each `.usda` file with the new access token.
3. Create a new Google Maps API Key in the [Cesium account](https://console.cloud.google.com/google/maps-apis/credentials?project=threed-tiles-api-testing).
    * The name of the key should match "Cesium for Omniverse Samples vX.X.X - Delete on April 1st, 2023" where the version is the same as the Cesium for Omniverse release and the expiry date is two months later than present.
    * The key should be restricted to the Map Tiles API.
4. For each `.usda` file using the Google 3D Tiles API, replace the key in the `cesium:url` property with the new key.
5. Verify that all the USD files load in Cesium for Omniverse.
6. Update `CHANGES.md`.
7. Commit the changes, e.g. `git commit -am "0.0.0 release"`.
8. Push the commit, e.g. `git push origin main`.
9. Tag the release, e.g. `git tag -a v0.0.0 -m "0.0.0 release"`.
10. Push the tag, e.g. `git push origin v0.0.0`.
11. Download the repo as a zip file.
12. Extract the zip file.
13. Rename the extracted folder, e.g. rename `cesium-omniverse-samples-main` to `CesiumOmniverseSamples-v0.0.0`.
14. Create a zip file of the folder
15. Create a new release on GitHub: https://github.com/CesiumGS/cesium-omniverse-samples/releases/new.
    * Choose the new tag.
    * Copy the changelog into the description. Follow the format used in previous releases.
    * Upload the zip file.


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
