import omni.usd
from pxr import Gf, UsdGeom, Usd
from datetime import datetime
from asyncio import ensure_future
from cesium.omniverse.api.globe_anchor import anchor_xform_at_path
from cesium.usd.plugins.CesiumUsdSchemas import CartographicPolygon
import logging


class CesiumCartographicPolygonUtility:
    _stage = None
    _up_axis = None
    _logger = None

    @classmethod
    def _prepare(cls):
        ctx = omni.usd.get_context()
        cls._stage = ctx.get_stage()
        cls._up_axis = UsdGeom.GetStageUpAxis(cls._stage)
        cls._logger = logging.getLogger(__name__)

    @classmethod
    def create_prim_footprints(cls, prim_list_str):
        """
        Creates UsdGeom.BasisCurves that match the footprint of each provided prim (including any child prims)
        The footprint is projected down on the current up axis.

        There must be at least one UsdGeom.Mesh in either the provided prim or its heirarchy of children

        Args:
            prim_path_list: A list of prim paths to create footprints from.
        Returns:
            A list of UsdGeom.BasisCurves prims created
        """
        cls._prepare()

        start_time = datetime.now()

        curves = []

        # Generate one polygon per mesh
        for mesh_path in prim_list_str:
            prim = cls._stage.GetPrimAtPath(mesh_path)

            up_value = 0.0  # TODO: Perhaps improve this - for now all shapes will be at 0.0

            # Find all the mesh prims for prim and its children
            mesh_prims = cls._get_mesh_prims(prim)

            # TODO: Sort the meshes by bounds large to small to potentially improve performance

            polygons = []

            # Find polygons for each mesh and add them to the current union
            for mesh_prim in mesh_prims:
                # Only process a mesh if bounds is outside the current polygons
                if not cls._is_prim_bounds_contained(mesh_prim, polygons):
                    polygons = cls._combine_polygons(polygons + cls._create_polygons_from_mesh(mesh_prim))

            # Convert the polygons to basis curves
            count = 0
            for polygon in polygons:
                polygon_path = f"/CesiumCartographicPolygons/{prim.GetName()}_polygon_{count}"
                curves.append(cls._convert_polygon_to_basis_curve(polygon, polygon_path, up_value))
                count += 1

        end_time = datetime.now()

        elapsed_time = end_time - start_time

        cls._logger.info(f"Footprint generation took {elapsed_time.total_seconds()} seconds")

        return curves

    @classmethod
    async def _convert(cls, prim_path_list):
        """
        Async function that creates CesiumCartographicPolygons for every UsdGeom.BasisCurves prim provided

        Args:
            prim_path_list: A list of prim paths to UsdGeom.BasisCurves
        Returns:
            None
        """
        cls._prepare()

        for curve_path in prim_path_list:
            curve_prim = cls._stage.GetPrimAtPath(curve_path)

            if curve_prim.GetTypeName() != "BasisCurves":
                continue

            polygon_path = curve_path + "_Cesium"

            if cls._stage.GetPrimAtPath(polygon_path).IsValid():
                cls._logger.warning(f"{polygon_path} already exists, skipping")
                continue

            # Create a new cartographic polygon
            polygon = CartographicPolygon.Define(cls._stage, polygon_path)
            polygon_prim = polygon.GetPrim()

            # Add a globe anchor
            anchor_xform_at_path(polygon_path)

            # Await after globe anchor, otherwise we'll experience a crash
            await omni.kit.app.get_app().next_update_async()

            # Iterate through the curve attributes and copy them to the new polygon
            curve_attributes = curve_prim.GetAttributes()
            for attrib in curve_attributes:
                value = attrib.Get()
                if value is not None:
                    polygon_prim.GetAttribute(attrib.GetName()).Set(attrib.Get())
                else:
                    polygon_prim.GetAttribute(attrib.GetName()).Clear()

    @classmethod
    def create_cartographic_polygons_from_curves(cls, prim_path_list):
        """
        Creates CesiumCartographicPolygons for every UsdGeom.BasisCurves prim provided

        Args:
            prim_path_list: A list of prim paths to UsdGeom.BasisCurves
        Returns:
            None
        """
        cls._prepare()

        # TODO: Update the below after globe anchor refactor
        ensure_future(cls._convert(prim_path_list))

    @classmethod
    def _flatten_vector(cls, vector: Gf.Vec3d):
        """
        Returns a Gf.Vec2d with the up-axis removed.  Up axis will depend on current stage settings

        Args:
            vector: A Gf.Vec3d to convert into a Gf.Vec2d
        Returns:
            A Gf.Vec2d
        """
        if cls._up_axis == "Z":
            return Gf.Vec2d(vector[0], vector[1])
        elif cls._up_axis == "Y":
            return Gf.Vec2d(vector[0], vector[2])
        else:
            raise ValueError("Invalid up_axis. Supported values are 'Y' or 'Z'.")

    @classmethod
    def _unflatten_vector(
        cls,
        vector: Gf.Vec2d,
        up_value=0.0,
    ):
        """
        Returns a Gf.Vec3d with the up-axis value.  Up axis will depend on current stage settings

        Args:
            vector: A Gf.Vec2d to convert into a Gf.Vec3d
        Returns:
            A Gf.Vec3d
        """
        if cls._up_axis == "Z":
            return Gf.Vec3d(vector[0], vector[1], up_value)
        elif cls._up_axis == "Y":
            return Gf.Vec3d(vector[0], up_value, vector[1])
        else:
            raise ValueError("Invalid up_axis. Supported values are 'Y' or 'Z'.")

    @classmethod
    def _combine_polygons(cls, polygons):
        """
        Combines multiple Shapely Polygons via unary union

        Args:
            polygons: A list of Shapely Polygons
        Returns:
            A list of Shapely Polygons
        """
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union

        # Use unary_union to merge the polygons into a concave hull
        result = unary_union(polygons)
        combined_polygons = []

        if type(result) == MultiPolygon:
            for poly in result.geoms:
                combined_polygons.append(Polygon(poly.exterior.coords))
        elif type(result) == Polygon:
            combined_polygons.append(Polygon(result.exterior.coords))

        return combined_polygons

    @classmethod
    def _create_polygons_from_mesh(cls, prim: UsdGeom.Mesh):
        """
        Creates one or more Shapely Polygons that match the outer boundary of the supplied prim on the up axis

        Args:
            prim: A UsdGeom.Mesh prim
        Returns:
            A list of Shapely Polygons
        """
        from shapely.geometry import Polygon

        polygons = []

        # Generate polygon from this prim
        if prim.IsA(UsdGeom.Mesh):
            world_transform: Gf.Matrix4d = omni.usd.get_world_transform_matrix(prim)

            # Transform the vertices into world space, then flatten
            vertices_2d = []
            for v in prim.GetAttribute("points").Get():
                vertices_2d.append(cls._flatten_vector(world_transform.Transform(Gf.Vec3d(v))))

            face_indices = prim.GetAttribute("faceVertexIndices").Get()

            polygons = []
            counter = 0
            # Iterate the faces and generate Shapely Polygons
            for vertex_count in prim.GetAttribute("faceVertexCounts").Get():
                face_coords = []

                # Iterate through all the indices in the face - not always 3
                for i in range(0, vertex_count):
                    face_coords.append(vertices_2d[face_indices[counter]])
                    counter += 1

                # Generate a shapely polygon
                polygon = Polygon(face_coords)

                # Only add valid polygons
                if polygon.is_valid:
                    polygons.append(Polygon(face_coords))

        return cls._combine_polygons(polygons)

    @classmethod
    def _convert_polygon_to_basis_curve(cls, polygon, path, up_value):
        """
        Creates a new UsdGeom.BasisCurves from a Shapely Polygon

        Args:
            polygon: A shapely polygon
            path: A stage path for the new UsdGeom.BasisCurves
            up_value: Up axis value to place the UsdGeom.BasisCurves at
        Returns:
            A UsdGeom.BasisCurves prim
        """
        curve: Usd.Prim = cls._stage.DefinePrim(path, "BasisCurves")

        width = 0.1 / UsdGeom.GetStageMetersPerUnit(cls._stage)  # Set the width of the curve

        points = []
        widths = []

        # Calculate the pivot based on the AABB and supplied up_value
        pivot = cls._unflatten_vector(Gf.Vec2d(polygon.centroid.x, polygon.centroid.y), up_value)

        # Iterate the polygon and create the points and widths array
        for p in polygon.exterior.coords:
            points.append(cls._unflatten_vector(p, up_value=up_value) - pivot)
            widths.append(width)

        # Set other attributes
        curve.GetAttribute("curveVertexCounts").Set([len(points)])
        curve.GetAttribute("points").Set(points)
        curve.GetAttribute("widths").Set(widths)
        curve.GetAttribute("purpose").Set("default")
        curve.GetAttribute("basis").Set("bezier")
        curve.GetAttribute("type").Set("linear")
        curve.GetAttribute("wrap").Set("periodic")

        xformable = UsdGeom.Xformable(curve)

        # Clear transforms
        xformable.SetXformOpOrder([])

        # Set Position
        xformable.AddTranslateOp().Set(value=pivot)

        return curve

    @classmethod
    def _get_mesh_prims(cls, parent_prim):
        """
        Returns a list of all UsdGeom.Mesh prims in the heirarchy of the provided prim and its children

        Args:
            parent_prim: A prim to find UsdGeom.Mesh prims under
        Returns:
            A list of UsdGeom.Mesh prims
        """
        mesh_prims = []

        if parent_prim.IsA(UsdGeom.Mesh):
            mesh_prims.append(parent_prim)

        for child_prim in parent_prim.GetAllChildren():
            mesh_prims += cls._get_mesh_prims(child_prim)

        return mesh_prims

    @staticmethod
    def _compute_bbox(prim: Usd.Prim) -> Gf.Range3d:
        """
        Compute Bounding Box using ComputeWorldBound at UsdGeom.Imageable
        See https://graphics.pixar.com/usd/release/api/class_usd_geom_imageable.html

        Args:
            prim: A prim to compute the bounding box.
        Returns:
            A range (i.e. bounding box), see more at: https://graphics.pixar.com/usd/release/api/class_gf_range3d.html
        """
        imageable = UsdGeom.Imageable(prim)
        time = Usd.TimeCode.Default()  # The time at which we compute the bounding box
        bound = imageable.ComputeWorldBound(time, UsdGeom.Tokens.default_)
        bound_range = bound.ComputeAlignedBox()
        return bound_range

    @classmethod
    def _create_bounds_polygon(cls, prim):
        """
        Creates a Shapely 2D polygon that matches the horizontal bounds of the prim

        Args:
            prim: A prim to compute the bounding box.
            up_axis: The up_axis to use in the bounds calculation
        Returns:
            A Shapely 2D Polygon
        """
        from shapely.geometry import Polygon

        bounds = cls._compute_bbox(prim)
        bounds_min_2d = cls._flatten_vector(bounds.min)
        bounds_max_2d = cls._flatten_vector(bounds.max)

        v1 = (bounds_min_2d[0], bounds_min_2d[1])
        v2 = (bounds_min_2d[0], bounds_max_2d[1])
        v3 = (bounds_max_2d[0], bounds_max_2d[1])
        v4 = (bounds_max_2d[0], bounds_min_2d[1])

        return Polygon([v1, v2, v3, v4])

    @classmethod
    def _is_prim_bounds_contained(cls, prim, polygons):
        """
        Returns True if the bounds of the prim is completely within any of the provided Shapely polygons

        Args:
            prim: A prim to check bounds
            up_axis: The up_axis to use in the bounds calculation
        Returns:
            A Shapely 2D Polygon
        """
        bounds_polygon = cls._create_bounds_polygon(prim)
        for c in polygons:
            if c.contains(bounds_polygon):
                return True
        return False
