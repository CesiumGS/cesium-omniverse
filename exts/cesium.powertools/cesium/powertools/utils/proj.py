import pyproj

ECEF_EPSG = 4978
WGS84_EPSG = 4979


def get_crs_name_from_epsg(epsg_code):
    try:
        return pyproj.crs.CRS.from_epsg(epsg_code).name
    except pyproj.exceptions.CRSError:
        return None


def epsg_to_epsg(from_epsg, to_epsg, x, y, z):
    from_crs = pyproj.crs.CRS.from_epsg(from_epsg)
    to_crs = pyproj.crs.CRS.from_epsg(to_epsg)

    # Define the from and to coordinate systems
    transformer = pyproj.Transformer.from_crs(from_crs, to_crs)

    # Convert ECEF coordinates to longitude, latitude, and height
    lat, lon, height = transformer.transform(x, y, z)

    return lat, lon, height


def epsg_to_wgs84(from_epsg, x, y, z):
    return epsg_to_epsg(from_epsg, WGS84_EPSG, x, y, z)


def epsg_to_ecef(from_epsg, x, y, z):
    return epsg_to_epsg(from_epsg, ECEF_EPSG, x, y, z)


def ecef_to_wgs84(x, y, z):
    return epsg_to_epsg(ECEF_EPSG, WGS84_EPSG, x, y, z)


def wgs84_to_ecef(x, y, z):
    return epsg_to_epsg(WGS84_EPSG, ECEF_EPSG, x, y, z)
