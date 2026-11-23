#pragma once

#include "cesium/omniverse/UsdTokens.h"

#include <carb/Numeric.h>
#include <omni/fabric/FabricUSD.h>

// clang-format off
#ifdef CESIUM_OMNI_MSVC
#pragma warning(push)
#pragma warning(disable : 4003)
#endif

// Macros for defining FabricTokens from each UsdTokens struct
// Lazy initialization to avoid calling Token::createImmortal() during static initialization
// before Fabric is ready
#define FABRIC_DEFINE_TOKEN_ELEM_1(elem) \
    const omni::fabric::Token& elem() { \
        static omni::fabric::Token token = omni::fabric::Token::createImmortal(pxr::UsdTokens1->elem.GetText()); \
        return token; \
    }

#define FABRIC_DEFINE_TOKEN_ELEM_2(elem) \
    const omni::fabric::Token& elem() { \
        static omni::fabric::Token token = omni::fabric::Token::createImmortal(pxr::UsdTokens2->elem.GetText()); \
        return token; \
    }

#define FABRIC_DEFINE_TOKEN_ELEM_3(elem) \
    const omni::fabric::Token& elem() { \
        static omni::fabric::Token token = omni::fabric::Token::createImmortal(pxr::UsdTokens3->elem.GetText()); \
        return token; \
    }

#define FABRIC_DEFINE_TOKEN_1(r, elem) \
    TF_PP_TUPLE_ELEM(0, _TF_PP_IFF(TF_PP_IS_TUPLE(elem), \
        (FABRIC_DEFINE_TOKEN_ELEM_1(TF_PP_TUPLE_ELEM(0, elem))), \
        (FABRIC_DEFINE_TOKEN_ELEM_1(elem))))

#define FABRIC_DEFINE_TOKEN_2(r, elem) \
    TF_PP_TUPLE_ELEM(0, _TF_PP_IFF(TF_PP_IS_TUPLE(elem), \
        (FABRIC_DEFINE_TOKEN_ELEM_2(TF_PP_TUPLE_ELEM(0, elem))), \
        (FABRIC_DEFINE_TOKEN_ELEM_2(elem))))

#define FABRIC_DEFINE_TOKEN_3(r, elem) \
    TF_PP_TUPLE_ELEM(0, _TF_PP_IFF(TF_PP_IS_TUPLE(elem), \
        (FABRIC_DEFINE_TOKEN_ELEM_3(TF_PP_TUPLE_ELEM(0, elem))), \
        (FABRIC_DEFINE_TOKEN_ELEM_3(elem))))

#define FABRIC_DEFINE_TOKENS_1(seq) TF_PP_SEQ_FOR_EACH(FABRIC_DEFINE_TOKEN_1, ~, seq)
#define FABRIC_DEFINE_TOKENS_2(seq) TF_PP_SEQ_FOR_EACH(FABRIC_DEFINE_TOKEN_2, ~, seq)
#define FABRIC_DEFINE_TOKENS_3(seq) TF_PP_SEQ_FOR_EACH(FABRIC_DEFINE_TOKEN_3, ~, seq)

#define FABRIC_DECLARE_TOKEN_ELEM(elem) const omni::fabric::Token& elem();

#define FABRIC_DECLARE_TOKEN(r, elem) \
    TF_PP_TUPLE_ELEM(0, _TF_PP_IFF(TF_PP_IS_TUPLE(elem), \
        (FABRIC_DECLARE_TOKEN_ELEM(TF_PP_TUPLE_ELEM(0, elem))), \
        (FABRIC_DECLARE_TOKEN_ELEM(elem))))

#define FABRIC_DECLARE_TOKENS(seq) TF_PP_SEQ_FOR_EACH(FABRIC_DECLARE_TOKEN, ~, seq)

#ifdef CESIUM_OMNI_MSVC
#pragma warning(pop)
#endif

namespace cesium::omniverse::FabricTokens {
FABRIC_DECLARE_TOKENS(USD_TOKENS_1);
FABRIC_DECLARE_TOKENS(USD_TOKENS_2);
FABRIC_DECLARE_TOKENS(USD_TOKENS_3);

const omni::fabric::Token feature_id_n(uint64_t index);
const omni::fabric::Token raster_overlay_n(uint64_t index);
const omni::fabric::Token inputs_raster_overlay_n(uint64_t index);
const omni::fabric::Token primvars_st_n(uint64_t index);
const omni::fabric::Token primvars_st_interpolation_n(uint64_t index);
const omni::fabric::Token property_n(uint64_t index);

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
