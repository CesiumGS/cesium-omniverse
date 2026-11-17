#pragma once

#ifdef CESIUM_OMNI_MSVC
#pragma warning(push)
#pragma warning(disable : 4244 4305 4100 4201 4127 4251)
#endif

#include <pxr/base/tf/staticTokens.h>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(pop)
#endif

// clang-format off
PXR_NAMESPACE_OPEN_SCOPE

#ifdef CESIUM_OMNI_MSVC
#pragma warning(push)
#pragma warning(disable : 4003)
#endif

// Note: variable names should match the USD token names as closely as possible, with special characters converted to underscores
// Split into multiple sequences to avoid MSVC C1009 "macros nested too deeply" error

#define USD_TOKENS_1 \
    (base_color_texture) \
    (cesium_base_color_texture_float4) \
    (cesium_feature_id_int) \
    (cesium_raster_overlay_float4) \
    (cesium_internal_clipping_raster_overlay_resolver) \
    (cesium_internal_feature_id_attribute_lookup) \
    (cesium_internal_feature_id_texture_lookup) \
    (cesium_internal_raster_overlay_lookup) \
    (cesium_internal_raster_overlay_resolver) \
    (cesium_internal_material) \
    (cesium_internal_property_attribute_float_lookup) \
    (cesium_internal_property_attribute_float2_lookup) \
    (cesium_internal_property_attribute_float3_lookup) \
    (cesium_internal_property_attribute_float4_lookup) \
    (cesium_internal_property_attribute_int_lookup) \
    (cesium_internal_property_attribute_int2_lookup) \
    (cesium_internal_property_attribute_int3_lookup) \
    (cesium_internal_property_attribute_int4_lookup) \
    (cesium_internal_property_attribute_normalized_int_lookup) \
    (cesium_internal_property_attribute_normalized_int2_lookup) \
    (cesium_internal_property_attribute_normalized_int3_lookup) \
    (cesium_internal_property_attribute_normalized_int4_lookup) \
    (cesium_internal_property_table_float_lookup) \
    (cesium_internal_property_table_float2_lookup) \
    (cesium_internal_property_table_float3_lookup) \
    (cesium_internal_property_table_float4_lookup) \
    (cesium_internal_property_table_int_lookup) \
    (cesium_internal_property_table_int2_lookup) \
    (cesium_internal_property_table_int3_lookup) \
    (cesium_internal_property_table_int4_lookup) \
    (cesium_internal_property_table_normalized_int_lookup) \
    (cesium_internal_property_table_normalized_int2_lookup) \
    (cesium_internal_property_table_normalized_int3_lookup) \
    (cesium_internal_property_table_normalized_int4_lookup) \
    (cesium_internal_property_texture_int_lookup) \
    (cesium_internal_property_texture_int2_lookup) \
    (cesium_internal_property_texture_int3_lookup) \
    (cesium_internal_property_texture_int4_lookup) \
    (cesium_internal_property_texture_normalized_int_lookup) \
    (cesium_internal_property_texture_normalized_int2_lookup) \
    (cesium_internal_property_texture_normalized_int3_lookup) \
    (cesium_internal_property_texture_normalized_int4_lookup) \
    (cesium_internal_texture_lookup) \
    (cesium_property_float) \
    (cesium_property_float2) \
    (cesium_property_float3) \
    (cesium_property_float4) \
    (cesium_property_int) \
    (cesium_property_int2) \
    (cesium_property_int3) \
    (cesium_property_int4)

#define USD_TOKENS_2 \
    (clipping_raster_overlay_resolver) \
    (doubleSided) \
    (extent) \
    (faceVertexCounts) \
    (faceVertexIndices) \
    (raster_overlay) \
    (raster_overlay_resolver) \
    (Material) \
    (Mesh) \
    (none) \
    (normals) \
    (points) \
    (Shader) \
    (sourceAsset) \
    (subdivisionScheme) \
    (vertex) \
    (vertexId) \
    (_cesium_gltfLocalToEcefTransform) \
    (_cesium_tilesetId) \
    (_deletedPrims) \
    (_materialSource) \
    (_paramColorSpace) \
    (_sdrMetadata) \
    (_worldExtent) \
    (_worldVisibility) \
    ((info_implementationSource, "info:implementationSource")) \
    ((info_mdl_sourceAsset, "info:mdl:sourceAsset")) \
    ((info_mdl_sourceAsset_subIdentifier, "info:mdl:sourceAsset:subIdentifier")) \
    ((inputs_alpha, "inputs:alpha")) \
    ((inputs_alpha_clip, "inputs:alpha_clip")) \
    ((inputs_alpha_cutoff, "inputs:alpha_cutoff")) \
    ((inputs_alpha_mode, "inputs:alpha_mode")) \
    ((inputs_base_alpha, "inputs:base_alpha")) \
    ((inputs_base_color_factor, "inputs:base_color_factor")) \
    ((inputs_base_color_texture, "inputs:base_color_texture")) \
    ((inputs_channels, "inputs:channels")) \
    ((inputs_channel_count, "inputs:channel_count")) \
    ((inputs_default_value, "inputs:default_value")) \
    ((inputs_emissive_factor, "inputs:emissive_factor")) \
    ((inputs_excludeFromWhiteMode, "inputs:excludeFromWhiteMode")) \
    ((inputs_feature_id, "inputs:feature_id")) \
    ((inputs_feature_id_set_index, "inputs:feature_id_set_index")) \
    ((inputs_has_no_data, "inputs:has_no_data")) \
    ((inputs_raster_overlay, "inputs:raster_overlay")) \
    ((inputs_raster_overlay_count, "inputs:raster_overlay_count")) \
    ((inputs_raster_overlay_index, "inputs:raster_overlay_index")) \
    ((inputs_maximum_value, "inputs:maximum_value")) \
    ((inputs_metallic_factor, "inputs:metallic_factor")) \
    ((inputs_no_data, "inputs:no_data")) \
    ((inputs_null_feature_id, "inputs:null_feature_id"))

#define USD_TOKENS_3 \
    ((inputs_offset, "inputs:offset")) \
    ((inputs_primvar_name, "inputs:primvar_name")) \
    ((inputs_property_id, "inputs:property_id")) \
    ((inputs_property_table_texture, "inputs:property_table_texture")) \
    ((inputs_property_value, "inputs:property_value")) \
    ((inputs_roughness_factor, "inputs:roughness_factor")) \
    ((inputs_scale, "inputs:scale")) \
    ((inputs_texture, "inputs:texture")) \
    ((inputs_tex_coord_index, "inputs:tex_coord_index")) \
    ((inputs_tex_coord_offset, "inputs:tex_coord_offset")) \
    ((inputs_tex_coord_rotation, "inputs:tex_coord_rotation")) \
    ((inputs_tex_coord_scale, "inputs:tex_coord_scale")) \
    ((inputs_tile_color, "inputs:tile_color")) \
    ((inputs_wrap_s, "inputs:wrap_s")) \
    ((inputs_wrap_t, "inputs:wrap_t")) \
    ((material_binding, "material:binding")) \
    ((normals_interpolation, "normals:interpolation")) \
    ((omni_fabric_localMatrix, "omni:fabric:localMatrix")) \
    ((omni_fabric_worldMatrix, "omni:fabric:worldMatrix")) \
    ((outputs_mdl_displacement, "outputs:mdl:displacement")) \
    ((outputs_mdl_surface, "outputs:mdl:surface")) \
    ((outputs_mdl_volume, "outputs:mdl:volume")) \
    ((outputs_out, "outputs:out")) \
    ((primvars_COLOR_0, "primvars:COLOR_0")) \
    ((primvars_COLOR_0_interpolation, "primvars:COLOR_0:interpolation")) \
    ((primvars_displayColor, "primvars:displayColor")) \
    ((primvars_displayOpacity, "primvars:displayOpacity")) \
    ((primvars_vertexId, "primvars:vertexId")) \
    ((primvars_vertexId_interpolation, "primvars:vertexId:interpolation")) \
    ((xformOp_orient, "xformOp:orient")) \
    ((xformOp_rotateXYZ, "xformOp:rotateXYZ")) \
    ((xformOp_rotateXZY, "xformOp:rotateXZY")) \
    ((xformOp_rotateYXZ, "xformOp:rotateYXZ")) \
    ((xformOp_rotateYZX, "xformOp:rotateYZX")) \
    ((xformOp_rotateZXY, "xformOp:rotateZXY")) \
    ((xformOp_rotateZYX, "xformOp:rotateZYX")) \
    ((xformOp_scale, "xformOp:scale")) \
    ((xformOp_translate, "xformOp:translate")) \
    ((_auto, "auto"))

#ifdef CESIUM_OMNI_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif

TF_DECLARE_PUBLIC_TOKENS(UsdTokens1, USD_TOKENS_1);
TF_DECLARE_PUBLIC_TOKENS(UsdTokens2, USD_TOKENS_2);
TF_DECLARE_PUBLIC_TOKENS(UsdTokens3, USD_TOKENS_3);

#ifdef CESIUM_OMNI_CLANG
#pragma clang diagnostic pop
#endif

PXR_NAMESPACE_CLOSE_SCOPE
// clang-format on
