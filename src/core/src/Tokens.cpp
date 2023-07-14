#include "cesium/omniverse/Tokens.h"

#include <omni/fabric/FabricUSD.h>
#include <pxr/base/tf/staticTokens.h>

// clang-format off
namespace pxr {

// Note: variable names should match the USD token names as closely as possible, with special characters converted to underscores

#ifdef CESIUM_OMNI_MSVC
__pragma(warning(push))
__pragma(warning(disable: 4003))
#endif

TF_DEFINE_PRIVATE_TOKENS(
    UsdTokens,
    (baseColorTex)
    (constant)
    (doubleSided)
    (extent)
    (faceVertexCounts)
    (faceVertexIndices)
    (gltf_material)
    ((gltf_pbr_mdl, "gltf/pbr.mdl"))
    (gltf_texture_lookup)
    (Material)
    (Mesh)
    (none)
    (out)
    (points)
    (primvarInterpolations)
    (primvars)
    (Shader)
    (sourceAsset)
    (subdivisionScheme)
    (vertex)
    (vertexColor)
    (wrap_s)
    (wrap_t)
    ((xformOp_translate, "xformOp:translate"))
    ((xformOp_rotation, "xformOp:rotation"))
    ((xformOp_scale, "xformOp:scale"))
    (_cesium_localToEcefTransform)
    (_cesium_tilesetId)
    (_deletedPrims)
    (_paramColorSpace)
    (_sdrMetadata)
    (_worldExtent)
    (_worldOrientation)
    (_worldPosition)
    (_worldScale)
    (_worldVisibility)
    ((_auto, "auto"))
    ((info_implementationSource, "info:implementationSource"))
    ((info_mdl_sourceAsset, "info:mdl:sourceAsset"))
    ((info_mdl_sourceAsset_subIdentifier, "info:mdl:sourceAsset:subIdentifier"))
    ((inputs_alpha_cutoff, "inputs:alpha_cutoff"))
    ((inputs_alpha_mode, "inputs:alpha_mode"))
    ((inputs_base_alpha, "inputs:base_alpha"))
    ((inputs_base_color_factor, "inputs:base_color_factor"))
    ((inputs_base_color_texture, "inputs:base_color_texture"))
    ((inputs_emissive_factor, "inputs:emissive_factor"))
    ((inputs_metallic_factor, "inputs:metallic_factor"))
    ((inputs_offset, "inputs:offset"))
    ((inputs_rotation, "inputs:rotation"))
    ((inputs_roughness_factor, "inputs:roughness_factor"))
    ((inputs_scale, "inputs:scale"))
    ((inputs_tex_coord_index, "inputs:tex_coord_index"))
    ((inputs_texture, "inputs:texture"))
    ((inputs_vertex_color_name, "inputs:vertex_color_name"))
    ((inputs_wrap_s, "inputs:wrap_s"))
    ((inputs_wrap_t, "inputs:wrap_t"))
    ((material_binding, "material:binding"))
    ((outputs_mdl_displacement, "outputs:mdl:displacement"))
    ((outputs_mdl_surface, "outputs:mdl:surface"))
    ((outputs_mdl_volume, "outputs:mdl:volume"))
    ((outputs_out, "outputs:out"))
    ((primvars_displayColor, "primvars:displayColor"))
    ((primvars_displayOpacity, "primvars:displayOpacity"))
    ((primvars_normals, "primvars:normals"))
    ((primvars_st, "primvars:st"))
    ((primvars_vertexColor, "primvars:vertexColor"))
);

#ifdef CESIUM_OMNI_MSVC
__pragma(warning(pop))
#endif

}

namespace cesium::omniverse::FabricTokens {
const omni::fabric::TokenC baseColorTex = omni::fabric::asInt(pxr::UsdTokens->baseColorTex);
const omni::fabric::TokenC constant = omni::fabric::asInt(pxr::UsdTokens->constant);
const omni::fabric::TokenC doubleSided = omni::fabric::asInt(pxr::UsdTokens->doubleSided);
const omni::fabric::TokenC extent = omni::fabric::asInt(pxr::UsdTokens->extent);
const omni::fabric::TokenC faceVertexCounts = omni::fabric::asInt(pxr::UsdTokens->faceVertexCounts);
const omni::fabric::TokenC faceVertexIndices = omni::fabric::asInt(pxr::UsdTokens->faceVertexIndices);
const omni::fabric::TokenC gltf_material = omni::fabric::asInt(pxr::UsdTokens->gltf_material);
const omni::fabric::TokenC gltf_pbr_mdl = omni::fabric::asInt(pxr::UsdTokens->gltf_pbr_mdl);
const omni::fabric::TokenC gltf_texture_lookup = omni::fabric::asInt(pxr::UsdTokens->gltf_texture_lookup);
const omni::fabric::TokenC info_implementationSource = omni::fabric::asInt(pxr::UsdTokens->info_implementationSource);
const omni::fabric::TokenC info_mdl_sourceAsset = omni::fabric::asInt(pxr::UsdTokens->info_mdl_sourceAsset);
const omni::fabric::TokenC info_mdl_sourceAsset_subIdentifier = omni::fabric::asInt(pxr::UsdTokens->info_mdl_sourceAsset_subIdentifier);
const omni::fabric::TokenC inputs_alpha_cutoff = omni::fabric::asInt(pxr::UsdTokens->inputs_alpha_cutoff);
const omni::fabric::TokenC inputs_alpha_mode = omni::fabric::asInt(pxr::UsdTokens->inputs_alpha_mode);
const omni::fabric::TokenC inputs_base_alpha = omni::fabric::asInt(pxr::UsdTokens->inputs_base_alpha);
const omni::fabric::TokenC inputs_base_color_factor = omni::fabric::asInt(pxr::UsdTokens->inputs_base_color_factor);
const omni::fabric::TokenC inputs_base_color_texture = omni::fabric::asInt(pxr::UsdTokens->inputs_base_color_texture);
const omni::fabric::TokenC inputs_emissive_factor = omni::fabric::asInt(pxr::UsdTokens->inputs_emissive_factor);
const omni::fabric::TokenC inputs_metallic_factor = omni::fabric::asInt(pxr::UsdTokens->inputs_metallic_factor);
const omni::fabric::TokenC inputs_offset = omni::fabric::asInt(pxr::UsdTokens->inputs_offset);
const omni::fabric::TokenC inputs_rotation = omni::fabric::asInt(pxr::UsdTokens->inputs_rotation);
const omni::fabric::TokenC inputs_roughness_factor = omni::fabric::asInt(pxr::UsdTokens->inputs_roughness_factor);
const omni::fabric::TokenC inputs_scale = omni::fabric::asInt(pxr::UsdTokens->inputs_scale);
const omni::fabric::TokenC inputs_tex_coord_index = omni::fabric::asInt(pxr::UsdTokens->inputs_tex_coord_index);
const omni::fabric::TokenC inputs_texture = omni::fabric::asInt(pxr::UsdTokens->inputs_texture);
const omni::fabric::TokenC inputs_vertex_color_name = omni::fabric::asInt(pxr::UsdTokens->inputs_vertex_color_name);
const omni::fabric::TokenC inputs_wrap_s = omni::fabric::asInt(pxr::UsdTokens->inputs_wrap_s);
const omni::fabric::TokenC inputs_wrap_t = omni::fabric::asInt(pxr::UsdTokens->inputs_wrap_t);
const omni::fabric::TokenC Material = omni::fabric::asInt(pxr::UsdTokens->Material);
const omni::fabric::TokenC material_binding = omni::fabric::asInt(pxr::UsdTokens->material_binding);
const omni::fabric::TokenC Mesh = omni::fabric::asInt(pxr::UsdTokens->Mesh);
const omni::fabric::TokenC none = omni::fabric::asInt(pxr::UsdTokens->none);
const omni::fabric::TokenC out = omni::fabric::asInt(pxr::UsdTokens->out);
const omni::fabric::TokenC outputs_mdl_displacement = omni::fabric::asInt(pxr::UsdTokens->outputs_mdl_displacement);
const omni::fabric::TokenC outputs_mdl_surface = omni::fabric::asInt(pxr::UsdTokens->outputs_mdl_surface);
const omni::fabric::TokenC outputs_mdl_volume = omni::fabric::asInt(pxr::UsdTokens->outputs_mdl_volume);
const omni::fabric::TokenC outputs_out = omni::fabric::asInt(pxr::UsdTokens->outputs_out);
const omni::fabric::TokenC points = omni::fabric::asInt(pxr::UsdTokens->points);
const omni::fabric::TokenC primvarInterpolations = omni::fabric::asInt(pxr::UsdTokens->primvarInterpolations);
const omni::fabric::TokenC primvars = omni::fabric::asInt(pxr::UsdTokens->primvars);
const omni::fabric::TokenC primvars_displayColor = omni::fabric::asInt(pxr::UsdTokens->primvars_displayColor);
const omni::fabric::TokenC primvars_displayOpacity = omni::fabric::asInt(pxr::UsdTokens->primvars_displayOpacity);
const omni::fabric::TokenC primvars_normals = omni::fabric::asInt(pxr::UsdTokens->primvars_normals);
const omni::fabric::TokenC primvars_st = omni::fabric::asInt(pxr::UsdTokens->primvars_st);
const omni::fabric::TokenC primvars_vertexColor = omni::fabric::asInt(pxr::UsdTokens->primvars_vertexColor);
const omni::fabric::TokenC Shader = omni::fabric::asInt(pxr::UsdTokens->Shader);
const omni::fabric::TokenC sourceAsset = omni::fabric::asInt(pxr::UsdTokens->sourceAsset);
const omni::fabric::TokenC subdivisionScheme = omni::fabric::asInt(pxr::UsdTokens->subdivisionScheme);
const omni::fabric::TokenC vertex = omni::fabric::asInt(pxr::UsdTokens->vertex);
const omni::fabric::TokenC vertexColor = omni::fabric::asInt(pxr::UsdTokens->vertexColor);
const omni::fabric::TokenC _auto = omni::fabric::asInt(pxr::UsdTokens->_auto);
const omni::fabric::TokenC _cesium_localToEcefTransform = omni::fabric::asInt(pxr::UsdTokens->_cesium_localToEcefTransform);
const omni::fabric::TokenC _cesium_tilesetId = omni::fabric::asInt(pxr::UsdTokens->_cesium_tilesetId);
const omni::fabric::TokenC _deletedPrims = omni::fabric::asInt(pxr::UsdTokens->_deletedPrims);
const omni::fabric::TokenC _paramColorSpace = omni::fabric::asInt(pxr::UsdTokens->_paramColorSpace);
const omni::fabric::TokenC _sdrMetadata = omni::fabric::asInt(pxr::UsdTokens->_sdrMetadata);
const omni::fabric::TokenC _worldExtent = omni::fabric::asInt(pxr::UsdTokens->_worldExtent);
const omni::fabric::TokenC _worldOrientation = omni::fabric::asInt(pxr::UsdTokens->_worldOrientation);
const omni::fabric::TokenC _worldPosition = omni::fabric::asInt(pxr::UsdTokens->_worldPosition);
const omni::fabric::TokenC _worldScale = omni::fabric::asInt(pxr::UsdTokens->_worldScale);
const omni::fabric::TokenC _worldVisibility = omni::fabric::asInt(pxr::UsdTokens->_worldVisibility);
}

namespace cesium::omniverse::UsdTokens {
const pxr::TfToken& baseColorTex = pxr::UsdTokens->baseColorTex;
const pxr::TfToken& constant = pxr::UsdTokens->constant;
const pxr::TfToken& doubleSided = pxr::UsdTokens->doubleSided;
const pxr::TfToken& extent = pxr::UsdTokens->extent;
const pxr::TfToken& faceVertexCounts = pxr::UsdTokens->faceVertexCounts;
const pxr::TfToken& faceVertexIndices = pxr::UsdTokens->faceVertexIndices;
const pxr::TfToken& gltf_material = pxr::UsdTokens->gltf_material;
const pxr::TfToken& gltf_pbr_mdl = pxr::UsdTokens->gltf_pbr_mdl;
const pxr::TfToken& gltf_texture_lookup = pxr::UsdTokens->gltf_texture_lookup;
const pxr::TfToken& info_implementationSource = pxr::UsdTokens->info_implementationSource;
const pxr::TfToken& info_mdl_sourceAsset = pxr::UsdTokens->info_mdl_sourceAsset;
const pxr::TfToken& info_mdl_sourceAsset_subIdentifier = pxr::UsdTokens->info_mdl_sourceAsset_subIdentifier;
const pxr::TfToken& inputs_alpha_cutoff = pxr::UsdTokens->inputs_alpha_cutoff;
const pxr::TfToken& inputs_alpha_mode = pxr::UsdTokens->inputs_alpha_mode;
const pxr::TfToken& inputs_base_alpha = pxr::UsdTokens->inputs_base_alpha;
const pxr::TfToken& inputs_base_color_factor = pxr::UsdTokens->inputs_base_color_factor;
const pxr::TfToken& inputs_base_color_texture = pxr::UsdTokens->inputs_base_color_texture;
const pxr::TfToken& inputs_emissive_factor = pxr::UsdTokens->inputs_emissive_factor;
const pxr::TfToken& inputs_metallic_factor = pxr::UsdTokens->inputs_metallic_factor;
const pxr::TfToken& inputs_offset = pxr::UsdTokens->inputs_offset;
const pxr::TfToken& inputs_rotation = pxr::UsdTokens->inputs_rotation;
const pxr::TfToken& inputs_roughness_factor = pxr::UsdTokens->inputs_roughness_factor;
const pxr::TfToken& inputs_scale = pxr::UsdTokens->inputs_scale;
const pxr::TfToken& inputs_tex_coord_index = pxr::UsdTokens->inputs_tex_coord_index;
const pxr::TfToken& inputs_texture = pxr::UsdTokens->inputs_texture;
const pxr::TfToken& inputs_vertex_color_name = pxr::UsdTokens->inputs_vertex_color_name;
const pxr::TfToken& inputs_wrap_s = pxr::UsdTokens->inputs_wrap_s;
const pxr::TfToken& inputs_wrap_t = pxr::UsdTokens->inputs_wrap_t;
const pxr::TfToken& Material = pxr::UsdTokens->Material;
const pxr::TfToken& material_binding = pxr::UsdTokens->material_binding;
const pxr::TfToken& Mesh = pxr::UsdTokens->Mesh;
const pxr::TfToken& none = pxr::UsdTokens->none;
const pxr::TfToken& out = pxr::UsdTokens->out;
const pxr::TfToken& outputs_mdl_displacement = pxr::UsdTokens->outputs_mdl_displacement;
const pxr::TfToken& outputs_mdl_surface = pxr::UsdTokens->outputs_mdl_surface;
const pxr::TfToken& outputs_mdl_volume = pxr::UsdTokens->outputs_mdl_volume;
const pxr::TfToken& outputs_out = pxr::UsdTokens->outputs_out;
const pxr::TfToken& points = pxr::UsdTokens->points;
const pxr::TfToken& primvarInterpolations = pxr::UsdTokens->primvarInterpolations;
const pxr::TfToken& primvars = pxr::UsdTokens->primvars;
const pxr::TfToken& primvars_displayColor = pxr::UsdTokens->primvars_displayColor;
const pxr::TfToken& primvars_displayOpacity = pxr::UsdTokens->primvars_displayOpacity;
const pxr::TfToken& primvars_normals = pxr::UsdTokens->primvars_normals;
const pxr::TfToken& primvars_st = pxr::UsdTokens->primvars_st;
const pxr::TfToken& primvars_vertexColor = pxr::UsdTokens->primvars_vertexColor;
const pxr::TfToken& Shader = pxr::UsdTokens->Shader;
const pxr::TfToken& sourceAsset = pxr::UsdTokens->sourceAsset;
const pxr::TfToken& subdivisionScheme = pxr::UsdTokens->subdivisionScheme;
const pxr::TfToken& vertex = pxr::UsdTokens->vertex;
const pxr::TfToken& vertexColor = pxr::UsdTokens->vertexColor;
const pxr::TfToken& wrap_s = pxr::UsdTokens->wrap_s;
const pxr::TfToken& wrap_t = pxr::UsdTokens->wrap_t;
const pxr::TfToken& xformOp_translate = pxr::UsdTokens->xformOp_translate;
const pxr::TfToken& xformOp_rotation = pxr::UsdTokens->xformOp_rotation;
const pxr::TfToken& xformOp_scale = pxr::UsdTokens->xformOp_scale;
const pxr::TfToken& _auto = pxr::UsdTokens->_auto;
const pxr::TfToken& _cesium_localToEcefTransform = pxr::UsdTokens->_cesium_localToEcefTransform;
const pxr::TfToken& _cesium_tilesetId = pxr::UsdTokens->_cesium_tilesetId;
const pxr::TfToken& _deletedPrims = pxr::UsdTokens->_deletedPrims;
const pxr::TfToken& _paramColorSpace = pxr::UsdTokens->_paramColorSpace;
const pxr::TfToken& _sdrMetadata = pxr::UsdTokens->_sdrMetadata;
const pxr::TfToken& _worldExtent = pxr::UsdTokens->_worldExtent;
const pxr::TfToken& _worldOrientation = pxr::UsdTokens->_worldOrientation;
const pxr::TfToken& _worldPosition = pxr::UsdTokens->_worldPosition;
const pxr::TfToken& _worldScale = pxr::UsdTokens->_worldScale;
const pxr::TfToken& _worldVisibility = pxr::UsdTokens->_worldVisibility;
}
// clang-format on
