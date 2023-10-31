# Releasing a new version of Cesium for Omniverse

This is the process we follow when releasing a new version of Cesium for Omniverse on GitHub.

1. [Release a new version of Cesium for Omniverse Samples](#releasing-a-new-version-of-cesium-for-omniverse-samples).
2. Make sure the latest commit in `main` is passing CI.
3. Download the latest build from S3. In the AWS management console (InternalServices AWS account), go to the bucket [`cesium-builds/cesium-omniverse/main`](https://s3.console.aws.amazon.com/s3/buckets/cesium-builds?region=us-east-1&prefix=cesium-omniverse/main/&showversions=false), find the appropriate date and commit hash to download the CentOS and Windows zip files (e.g. `CesiumGS-cesium-omniverse-linux-x86_64-xxxxxxx.zip` and `CesiumGS-cesium-omniverse-windows-x86_64-xxxxxxx.zip`)
4. Verify that the Linux package loads in USD Composer (see instructions below).
5. Verify that the Windows package loads in USD Composer (see instructions below).
6. Update the project `VERSION` in [CMakeLists.txt](../../CMakeLists.txt).
7. Update the extension `version` in [extension.toml](../../exts/cesium.omniverse/config/extension.toml). This should be the same version as above.
8. Update [`CHANGES.md`](../../CHANGES.md).
9. Update `ION_ACCESS_TOKEN` in [`extension.py`](../../apps/exts/cesium.performance.app/cesium/performance/app/extension.py) within `cesium.performance.app` using the newly generated keys.
10. Create a branch, e.g. `git checkout -b release-0.0.0`.
11. Commit the changes, e.g. `git commit -am "0.0.0 release"`.
12. Push the commit, e.g. `git push origin release-0.0.0`.
13. Open a PR and merge the branch with "Rebase and merge".
14. Tag the release, e.g. `git tag -a v0.0.0 -m "0.0.0 release"`.
15. Push the tag, e.g. `git push origin v0.0.0`.
16. Wait for CI to pass.
17. Download the latest build from S3. In the AWS management console (InternalServices AWS account), go to the bucket [`cesium-builds/cesium-omniverse`](https://s3.console.aws.amazon.com/s3/buckets/cesium-builds?prefix=cesium-omniverse/&region=us-east-1), find the folder with the new tag and download the CentOS and Windows zip files (e.g. `CesiumGS-cesium-omniverse-linux-x86_64-v0.0.0.zip` and `CesiumGS-cesium-omniverse-windows-x86_64-v0.0.0.zip` )
18. Create a new release on GitHub: https://github.com/CesiumGS/cesium-omniverse/releases/new.
    * Chose the new tag.
    * Copy the changelog into the description. Follow the format used in previous releases.
    * Upload the Linux and Windows release zip files.

# Releasing a new version of Cesium for Omniverse Samples

1. Create a new access token using the CesiumJS ion account.
    * The name of the token should match "Cesium for Omniverse Samples vX.X.X - Delete on April 1st, 2023" where the version is the same as the Cesium for Omniverse release and the expiry date is two months later than present.
    * The scope of the token should be "assets:read" for all assets.
2. Replace the `cesium:projectDefaultIonAccessToken` property in each `.usda` file with the new access token.
3. Verify that all the USD files load in Cesium for Omniverse.
4. Update `CHANGES.md`.
5. Commit the changes, e.g. `git commit -am "0.0.0 release"`.
6. Push the commit, e.g. `git push origin main`.
7. Tag the release, e.g. `git tag -a v0.0.0 -m "0.0.0 release"`.
8. Push the tag, e.g. `git push origin v0.0.0`.
9. Download the repo as a zip file.
10. Extract the zip file.
11. Rename the extracted folder, e.g. rename `cesium-omniverse-samples-main` to `CesiumOmniverseSamples-v0.0.0`.
12. Create a zip file of the folder
13. Create a new release on GitHub: https://github.com/CesiumGS/cesium-omniverse-samples/releases/new.
    * Choose the new tag.
    * Copy the changelog into the description. Follow the format used in previous releases.
    * Upload the zip file.


# Verify Package

After the package is built, verify that the extension loads in USD Composer:

* Open USD Composer
* Open the extensions window and remove Cesium for Omniverse from the list of search paths (if it exists)
* Close USD Composer
* Unzip the package to `$USERHOME$/Documents/Kit/Shared/exts`
* Open USD Composer
* Open the extensions window and enable autoload for Cesium for Omniverse
* Restart USD Composer
* Verify that there aren't any console errors
* Verify that you can load Cesium World Terrain and OSM buildings
* Delete the extensions from `$USERHOME$/Documents/Kit/Shared/exts`
