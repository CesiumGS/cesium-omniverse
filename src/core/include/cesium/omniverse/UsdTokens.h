#pragma once

#include <omni/fabric/FabricUSD.h>
#include <pxr/base/tf/staticTokens.h>

// clang-format off
PXR_NAMESPACE_OPEN_SCOPE

#ifdef CESIUM_OMNI_MSVC
__pragma(warning(push)) __pragma(warning(disable : 4003))
#endif

// Note: variable names should match the USD token names as closely as possible, with special characters converted to underscores

#define USD_TOKENS \
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
    (cesium_property_int4) \
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
    ((inputs_null_feature_id, "inputs:null_feature_id")) \
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

TF_DECLARE_PUBLIC_TOKENS(UsdTokens, USD_TOKENS);

#ifdef CESIUM_OMNI_CLANG
#pragma clang diagnostic pop
#endif

#define FABRIC_DEFINE_TOKEN_ELEM(elem) const omni::fabric::TokenC elem = omni::fabric::asInt(pxr::UsdTokens->elem);

#define FABRIC_DEFINE_TOKEN(r, data, elem) \
    BOOST_PP_TUPLE_ELEM(1, 0, BOOST_PP_IIF(TF_PP_IS_TUPLE(elem), \
        (FABRIC_DEFINE_TOKEN_ELEM(BOOST_PP_TUPLE_ELEM(2, 0, elem))), \
        (FABRIC_DEFINE_TOKEN_ELEM(elem))))

#define FABRIC_DEFINE_TOKENS(seq) BOOST_PP_SEQ_FOR_EACH(FABRIC_DEFINE_TOKEN, ~, seq)

#define FABRIC_DECLARE_TOKEN_ELEM(elem) extern const omni::fabric::TokenC elem;

#define FABRIC_DECLARE_TOKEN(r, data, elem) \
    BOOST_PP_TUPLE_ELEM(1, 0, BOOST_PP_IIF(TF_PP_IS_TUPLE(elem), \
        (FABRIC_DECLARE_TOKEN_ELEM(BOOST_PP_TUPLE_ELEM(2, 0, elem))), \
        (FABRIC_DECLARE_TOKEN_ELEM(elem))))

#define FABRIC_DECLARE_TOKENS(seq) BOOST_PP_SEQ_FOR_EACH(FABRIC_DECLARE_TOKEN, ~, seq)

#ifdef CESIUM_OMNI_MSVC
__pragma(warning(pop))
#endif

PXR_NAMESPACE_CLOSE_SCOPE

namespace cesium::omniverse::FabricTokens {
FABRIC_DECLARE_TOKENS(USD_TOKENS);

const omni::fabric::TokenC feature_id_n(uint64_t index);
const omni::fabric::TokenC raster_overlay_n(uint64_t index);
const omni::fabric::TokenC inputs_raster_overlay_n(uint64_t index);
const omni::fabric::TokenC primvars_st_n(uint64_t index);
const omni::fabric::TokenC primvars_st_interpolation_n(uint64_t index);
const omni::fabric::TokenC property_n(uint64_t index);

}

namespace cesium::omniverse::FabricTypes {
// Due to legacy support the eRelationship type is defined as a scalar value but is secretly an array
const omni::fabric::Type doubleSided(omni::fabric::BaseDataType::eBool, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type extent(omni::fabric::BaseDataType::eDouble, 6, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type faceVertexCounts(omni::fabric::BaseDataType::eInt, 1, 1, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type faceVertexIndices(omni::fabric::BaseDataType::eInt, 1, 1, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type info_implementationSource(omni::fabric::BaseDataType::eToken, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type info_mdl_sourceAsset(omni::fabric::BaseDataType::eAsset, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type info_mdl_sourceAsset_subIdentifier(omni::fabric::BaseDataType::eToken, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_alpha(omni::fabric::BaseDataType::eFloat, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_alpha_cutoff(omni::fabric::BaseDataType::eFloat, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_alpha_mode(omni::fabric::BaseDataType::eInt, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_base_alpha(omni::fabric::BaseDataType::eFloat, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_base_color_factor(omni::fabric::BaseDataType::eFloat, 3, 0, omni::fabric::AttributeRole::eColor);
const omni::fabric::Type inputs_channels(omni::fabric::BaseDataType::eInt, 4, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_channel_count(omni::fabric::BaseDataType::eInt, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_default_value_float(omni::fabric::BaseDataType::eFloat, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_default_value_float2(omni::fabric::BaseDataType::eFloat, 2, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_default_value_float3(omni::fabric::BaseDataType::eFloat, 3, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_default_value_float4(omni::fabric::BaseDataType::eFloat, 4, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_default_value_int(omni::fabric::BaseDataType::eInt, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_default_value_int2(omni::fabric::BaseDataType::eInt, 2, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_default_value_int3(omni::fabric::BaseDataType::eInt, 3, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_default_value_int4(omni::fabric::BaseDataType::eInt, 4, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_emissive_factor(omni::fabric::BaseDataType::eFloat, 3, 0, omni::fabric::AttributeRole::eColor);
const omni::fabric::Type inputs_excludeFromWhiteMode(omni::fabric::BaseDataType::eBool, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_has_no_data(omni::fabric::BaseDataType::eBool, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_raster_overlay_count(omni::fabric::BaseDataType::eInt, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_maximum_value_int(omni::fabric::BaseDataType::eInt, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_maximum_value_int2(omni::fabric::BaseDataType::eInt, 2, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_maximum_value_int3(omni::fabric::BaseDataType::eInt, 3, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_maximum_value_int4(omni::fabric::BaseDataType::eInt, 4, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_metallic_factor(omni::fabric::BaseDataType::eFloat, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_no_data_float(omni::fabric::BaseDataType::eFloat, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_no_data_float2(omni::fabric::BaseDataType::eFloat, 2, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_no_data_float3(omni::fabric::BaseDataType::eFloat, 3, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_no_data_float4(omni::fabric::BaseDataType::eFloat, 4, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_no_data_int(omni::fabric::BaseDataType::eInt, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_no_data_int2(omni::fabric::BaseDataType::eInt, 2, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_no_data_int3(omni::fabric::BaseDataType::eInt, 3, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_no_data_int4(omni::fabric::BaseDataType::eInt, 4, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_null_feature_id(omni::fabric::BaseDataType::eInt, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_offset_float(omni::fabric::BaseDataType::eFloat, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_offset_float2(omni::fabric::BaseDataType::eFloat, 2, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_offset_float3(omni::fabric::BaseDataType::eFloat, 3, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_offset_float4(omni::fabric::BaseDataType::eFloat, 4, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_primvar_name(omni::fabric::BaseDataType::eUChar, 1, 1, omni::fabric::AttributeRole::eText);
const omni::fabric::Type inputs_property_table_texture(omni::fabric::BaseDataType::eAsset, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_roughness_factor(omni::fabric::BaseDataType::eFloat, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_scale_float(omni::fabric::BaseDataType::eFloat, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_scale_float2(omni::fabric::BaseDataType::eFloat, 2, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_scale_float3(omni::fabric::BaseDataType::eFloat, 3, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_scale_float4(omni::fabric::BaseDataType::eFloat, 4, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_texture(omni::fabric::BaseDataType::eAsset, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_tex_coord_index(omni::fabric::BaseDataType::eInt, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_tex_coord_offset(omni::fabric::BaseDataType::eFloat, 2, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_tex_coord_rotation(omni::fabric::BaseDataType::eFloat, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_tex_coord_scale(omni::fabric::BaseDataType::eFloat, 2, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_tile_color(omni::fabric::BaseDataType::eFloat, 4, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_wrap_s(omni::fabric::BaseDataType::eInt, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_wrap_t(omni::fabric::BaseDataType::eInt, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type Material(omni::fabric::BaseDataType::eTag, 1, 0, omni::fabric::AttributeRole::ePrimTypeName);
const omni::fabric::Type material_binding(omni::fabric::BaseDataType::eRelationship, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type Mesh(omni::fabric::BaseDataType::eTag, 1, 0, omni::fabric::AttributeRole::ePrimTypeName);
const omni::fabric::Type normals(omni::fabric::BaseDataType::eFloat, 3, 1, omni::fabric::AttributeRole::eNormal);
const omni::fabric::Type omni_fabric_localMatrix(omni::fabric::BaseDataType::eDouble, 16, 0, omni::fabric::AttributeRole::eMatrix);
const omni::fabric::Type omni_fabric_worldMatrix(omni::fabric::BaseDataType::eDouble, 16, 0, omni::fabric::AttributeRole::eMatrix);
const omni::fabric::Type outputs_out(omni::fabric::BaseDataType::eToken, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type points(omni::fabric::BaseDataType::eFloat, 3, 1, omni::fabric::AttributeRole::ePosition);
const omni::fabric::Type primvars_COLOR_0(omni::fabric::BaseDataType::eFloat, 4, 1, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type primvars_interpolation(omni::fabric::BaseDataType::eToken, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type primvars_st(omni::fabric::BaseDataType::eFloat, 2, 1, omni::fabric::AttributeRole::eTexCoord);
const omni::fabric::Type primvars_vertexId(omni::fabric::BaseDataType::eFloat, 1, 1, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type Shader(omni::fabric::BaseDataType::eTag, 1, 0, omni::fabric::AttributeRole::ePrimTypeName);
const omni::fabric::Type subdivisionScheme(omni::fabric::BaseDataType::eToken, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type _cesium_gltfLocalToEcefTransform(omni::fabric::BaseDataType::eDouble, 16, 0, omni::fabric::AttributeRole::eMatrix);
const omni::fabric::Type _cesium_tilesetId(omni::fabric::BaseDataType::eInt64, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type _materialSource(omni::fabric::BaseDataType::eRelationship, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type _paramColorSpace(omni::fabric::BaseDataType::eToken, 1, 1, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type _sdrMetadata(omni::fabric::BaseDataType::eToken, 1, 1, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type _worldExtent(omni::fabric::BaseDataType::eDouble, 6, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type _worldVisibility(omni::fabric::BaseDataType::eBool, 1, 0, omni::fabric::AttributeRole::eNone);
}
// clang-format on
