#pragma once

#include <omni/fabric/FabricUSD.h>
#include <pxr/base/tf/staticTokens.h>

// clang-format off
namespace pxr {

#ifdef CESIUM_OMNI_MSVC
__pragma(warning(push)) __pragma(warning(disable : 4003))
#endif

// Note: variable names should match the USD token names as closely as possible, with special characters converted to underscores

#define USD_TOKENS \
    (base_color_texture) \
    (cesium) \
    (cesium_imagery_layer_resolver) \
    (cesium_material) \
    (cesium_texture_lookup) \
    (constant) \
    (doubleSided) \
    (extent) \
    (faceVertexCounts) \
    (faceVertexIndices) \
    (imagery_layer_0) \
    (imagery_layer_1) \
    (imagery_layer_2) \
    (imagery_layer_3) \
    (imagery_layer_4) \
    (imagery_layer_5) \
    (imagery_layer_6) \
    (imagery_layer_7) \
    (imagery_layer_8) \
    (imagery_layer_9) \
    (imagery_layer_10) \
    (imagery_layer_11) \
    (imagery_layer_12) \
    (imagery_layer_13) \
    (imagery_layer_14) \
    (imagery_layer_15) \
    (imagery_layer_resolver) \
    (Material) \
    (Mesh) \
    (none) \
    (points) \
    (primvarInterpolations) \
    (primvars) \
    (Shader) \
    (sourceAsset) \
    (subdivisionScheme) \
    (vertex) \
    (vertexColor) \
    (_cesium_localToEcefTransform) \
    (_cesium_tilesetId) \
    (_deletedPrims) \
    (_paramColorSpace) \
    (_sdrMetadata) \
    (_worldExtent) \
    (_worldOrientation) \
    (_worldPosition) \
    (_worldScale) \
    (_worldVisibility) \
    ((_auto, "auto")) \
    ((info_implementationSource, "info:implementationSource")) \
    ((info_mdl_sourceAsset, "info:mdl:sourceAsset")) \
    ((info_mdl_sourceAsset_subIdentifier, "info:mdl:sourceAsset:subIdentifier")) \
    ((inputs_alpha_cutoff, "inputs:alpha_cutoff")) \
    ((inputs_alpha_mode, "inputs:alpha_mode")) \
    ((inputs_base_alpha, "inputs:base_alpha")) \
    ((inputs_base_color_factor, "inputs:base_color_factor")) \
    ((inputs_base_color_texture, "inputs:base_color_texture")) \
    ((inputs_emissive_factor, "inputs:emissive_factor")) \
    ((inputs_excludeFromWhiteMode, "inputs:excludeFromWhiteMode")) \
    ((inputs_metallic_factor, "inputs:metallic_factor")) \
    ((inputs_offset, "inputs:offset")) \
    ((inputs_rotation, "inputs:rotation")) \
    ((inputs_roughness_factor, "inputs:roughness_factor")) \
    ((inputs_scale, "inputs:scale")) \
    ((inputs_tex_coord_index, "inputs:tex_coord_index")) \
    ((inputs_texture, "inputs:texture")) \
    ((inputs_imagery_layer_0, "inputs:imagery_layer_0")) \
    ((inputs_imagery_layer_1, "inputs:imagery_layer_1")) \
    ((inputs_imagery_layer_2, "inputs:imagery_layer_2")) \
    ((inputs_imagery_layer_3, "inputs:imagery_layer_3")) \
    ((inputs_imagery_layer_4, "inputs:imagery_layer_4")) \
    ((inputs_imagery_layer_5, "inputs:imagery_layer_5")) \
    ((inputs_imagery_layer_6, "inputs:imagery_layer_6")) \
    ((inputs_imagery_layer_7, "inputs:imagery_layer_7")) \
    ((inputs_imagery_layer_8, "inputs:imagery_layer_8")) \
    ((inputs_imagery_layer_9, "inputs:imagery_layer_9")) \
    ((inputs_imagery_layer_10, "inputs:imagery_layer_10")) \
    ((inputs_imagery_layer_11, "inputs:imagery_layer_11")) \
    ((inputs_imagery_layer_12, "inputs:imagery_layer_12")) \
    ((inputs_imagery_layer_13, "inputs:imagery_layer_13")) \
    ((inputs_imagery_layer_14, "inputs:imagery_layer_14")) \
    ((inputs_imagery_layer_15, "inputs:imagery_layer_15")) \
    ((inputs_imagery_layers_count, "inputs:imagery_layers_count")) \
    ((inputs_imagery_layers_texture, "inputs:imagery_layers_texture")) \
    ((inputs_vertex_color_name, "inputs:vertex_color_name")) \
    ((inputs_wrap_s, "inputs:wrap_s")) \
    ((inputs_wrap_t, "inputs:wrap_t")) \
    ((material_binding, "material:binding")) \
    ((outputs_mdl_displacement, "outputs:mdl:displacement")) \
    ((outputs_mdl_surface, "outputs:mdl:surface")) \
    ((outputs_mdl_volume, "outputs:mdl:volume")) \
    ((outputs_out, "outputs:out")) \
    ((primvars_displayColor, "primvars:displayColor")) \
    ((primvars_displayOpacity, "primvars:displayOpacity")) \
    ((primvars_normals, "primvars:normals")) \
    ((primvars_st_0, "primvars:st_0")) \
    ((primvars_st_1, "primvars:st_1")) \
    ((primvars_st_2, "primvars:st_2")) \
    ((primvars_st_3, "primvars:st_3")) \
    ((primvars_st_4, "primvars:st_4")) \
    ((primvars_st_5, "primvars:st_5")) \
    ((primvars_st_6, "primvars:st_6")) \
    ((primvars_st_7, "primvars:st_7")) \
    ((primvars_st_8, "primvars:st_8")) \
    ((primvars_st_9, "primvars:st_9")) \
    ((primvars_vertexColor, "primvars:vertexColor")) \
    ((xformOp_transform_cesium, "xformOp:transform:cesium"))

TF_DECLARE_PUBLIC_TOKENS(UsdTokens, USD_TOKENS);

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

}

namespace cesium::omniverse::FabricTokens {
FABRIC_DECLARE_TOKENS(USD_TOKENS);

const uint64_t MAX_PRIMVAR_ST_COUNT = 10;
const uint64_t MAX_IMAGERY_LAYERS_COUNT = 16;

const std::array<const omni::fabric::TokenC, MAX_PRIMVAR_ST_COUNT> primvars_st_n = {{
    primvars_st_0,
    primvars_st_1,
    primvars_st_2,
    primvars_st_3,
    primvars_st_4,
    primvars_st_5,
    primvars_st_6,
    primvars_st_7,
    primvars_st_8,
    primvars_st_9,
}};

const std::array<const omni::fabric::TokenC, MAX_IMAGERY_LAYERS_COUNT> imagery_layer_n = {{
    imagery_layer_0,
    imagery_layer_1,
    imagery_layer_2,
    imagery_layer_3,
    imagery_layer_4,
    imagery_layer_5,
    imagery_layer_6,
    imagery_layer_7,
    imagery_layer_8,
    imagery_layer_9,
    imagery_layer_10,
    imagery_layer_11,
    imagery_layer_12,
    imagery_layer_13,
    imagery_layer_14,
    imagery_layer_15,
}};

const std::array<const omni::fabric::TokenC, MAX_IMAGERY_LAYERS_COUNT> inputs_imagery_layer_n = {{
    inputs_imagery_layer_0,
    inputs_imagery_layer_1,
    inputs_imagery_layer_2,
    inputs_imagery_layer_3,
    inputs_imagery_layer_4,
    inputs_imagery_layer_5,
    inputs_imagery_layer_6,
    inputs_imagery_layer_7,
    inputs_imagery_layer_8,
    inputs_imagery_layer_9,
    inputs_imagery_layer_10,
    inputs_imagery_layer_11,
    inputs_imagery_layer_12,
    inputs_imagery_layer_13,
    inputs_imagery_layer_14,
    inputs_imagery_layer_15,
}};

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
const omni::fabric::Type inputs_alpha_cutoff(omni::fabric::BaseDataType::eFloat, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_alpha_mode(omni::fabric::BaseDataType::eInt, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_base_alpha(omni::fabric::BaseDataType::eFloat, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_base_color_factor(omni::fabric::BaseDataType::eFloat, 3, 0, omni::fabric::AttributeRole::eColor);
const omni::fabric::Type inputs_emissive_factor(omni::fabric::BaseDataType::eFloat, 3, 0, omni::fabric::AttributeRole::eColor);
const omni::fabric::Type inputs_excludeFromWhiteMode(omni::fabric::BaseDataType::eBool, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_metallic_factor(omni::fabric::BaseDataType::eFloat, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_offset(omni::fabric::BaseDataType::eFloat, 2, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_rotation(omni::fabric::BaseDataType::eFloat, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_roughness_factor(omni::fabric::BaseDataType::eFloat, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_scale(omni::fabric::BaseDataType::eFloat, 2, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_tex_coord_index(omni::fabric::BaseDataType::eInt, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_texture(omni::fabric::BaseDataType::eAsset, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_imagery_layers_count(omni::fabric::BaseDataType::eInt, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_vertex_color_name(omni::fabric::BaseDataType::eUChar, 1, 1, omni::fabric::AttributeRole::eText);
const omni::fabric::Type inputs_wrap_s(omni::fabric::BaseDataType::eInt, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type inputs_wrap_t(omni::fabric::BaseDataType::eInt, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type Material(omni::fabric::BaseDataType::eTag, 1, 0, omni::fabric::AttributeRole::ePrimTypeName);
const omni::fabric::Type material_binding(omni::fabric::BaseDataType::eRelationship, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type Mesh(omni::fabric::BaseDataType::eTag, 1, 0, omni::fabric::AttributeRole::ePrimTypeName);
const omni::fabric::Type outputs_out(omni::fabric::BaseDataType::eToken, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type points(omni::fabric::BaseDataType::eFloat, 3, 1, omni::fabric::AttributeRole::ePosition);
const omni::fabric::Type primvarInterpolations(omni::fabric::BaseDataType::eToken, 1, 1, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type primvars(omni::fabric::BaseDataType::eToken, 1, 1, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type primvars_displayColor(omni::fabric::BaseDataType::eFloat, 3, 1, omni::fabric::AttributeRole::eColor);
const omni::fabric::Type primvars_displayOpacity(omni::fabric::BaseDataType::eFloat, 1, 1, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type primvars_normals(omni::fabric::BaseDataType::eFloat, 3, 1, omni::fabric::AttributeRole::eNormal);
const omni::fabric::Type primvars_st(omni::fabric::BaseDataType::eFloat, 2, 1, omni::fabric::AttributeRole::eTexCoord);
const omni::fabric::Type primvars_vertexColor(omni::fabric::BaseDataType::eFloat, 3, 1, omni::fabric::AttributeRole::eColor);
const omni::fabric::Type Shader(omni::fabric::BaseDataType::eTag, 1, 0, omni::fabric::AttributeRole::ePrimTypeName);
const omni::fabric::Type subdivisionScheme(omni::fabric::BaseDataType::eToken, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type _cesium_localToEcefTransform(omni::fabric::BaseDataType::eDouble, 16, 0, omni::fabric::AttributeRole::eMatrix);
const omni::fabric::Type _cesium_tilesetId(omni::fabric::BaseDataType::eInt64, 1, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type _paramColorSpace(omni::fabric::BaseDataType::eToken, 1, 1, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type _sdrMetadata(omni::fabric::BaseDataType::eToken, 1, 1, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type _worldExtent(omni::fabric::BaseDataType::eDouble, 6, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type _worldOrientation(omni::fabric::BaseDataType::eFloat, 4, 0, omni::fabric::AttributeRole::eQuaternion);
const omni::fabric::Type _worldPosition(omni::fabric::BaseDataType::eDouble, 3, 0, omni::fabric::AttributeRole::eNone);
const omni::fabric::Type _worldScale(omni::fabric::BaseDataType::eFloat, 3, 0, omni::fabric::AttributeRole::eVector);
const omni::fabric::Type _worldVisibility(omni::fabric::BaseDataType::eBool, 1, 0, omni::fabric::AttributeRole::eNone);
}
// clang-format on
