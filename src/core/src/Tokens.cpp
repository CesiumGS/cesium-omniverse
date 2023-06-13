#include "cesium/omniverse/Tokens.h"

#include <carb/flatcache/FlatCacheUSD.h>
#include <pxr/base/tf/staticTokens.h>

// clang-format off
namespace pxr {

// Note: variable names should match the USD token names as closely as possible, with special characters converted to underscores
TF_DEFINE_PRIVATE_TOKENS(
    UsdTokens,
    (alpha_cutoff)
    (alpha_mode)
    (base_alpha)
    (base_color_factor)
    (base_color_texture)
    (baseColorTex)
    (constant)
    (displacement)
    (doubleSided)
    (emissive_factor)
    (faceVertexCounts)
    (faceVertexIndices)
    (gltf_material)
    (gltf_texture_lookup)
    (Material)
    (materialId)
    (MaterialNetwork)
    (Mesh)
    (metallic_factor)
    (none)
    (offset)
    (out)
    (points)
    (primvarInterpolations)
    (primvars)
    (rotation)
    (roughness_factor)
    (scale)
    (Shader)
    (subdivisionScheme)
    (surface)
    (tex_coord_index)
    (texture)
    (vertex)
    (vertex_color_name)
    (vertexColor)
    (wrap_s)
    (wrap_t)
    (_cesium_localToEcefTransform)
    (_cesium_tileId)
    (_cesium_tilesetId)
    (_deletedPrims)
    (_localExtent)
    (_nodePaths)
    (_paramColorSpace)
    (_parameters)
    (_relationships_inputId)
    (_relationships_inputName)
    (_relationships_outputId)
    (_relationships_outputName)
    (_terminals)
    (_worldExtent)
    (_worldOrientation)
    (_worldPosition)
    (_worldScale)
    (_worldVisibility)
    ((_auto, "auto"))
    ((gltf_pbr_mdl, "gltf/pbr.mdl"))
    ((info_id, "info:id"))
    ((info_sourceAsset_subIdentifier, "info:sourceAsset:subIdentifier"))
    ((primvars_displayColor, "primvars:displayColor"))
    ((primvars_displayOpacity, "primvars:displayOpacity"))
    ((primvars_normals, "primvars:normals"))
    ((primvars_st, "primvars:st"))
    ((primvars_vertexColor, "primvars:vertexColor"))
);
}

namespace cesium::omniverse::FabricTokens {
const carb::flatcache::TokenC alpha_cutoff = carb::flatcache::asInt(pxr::UsdTokens->alpha_cutoff);
const carb::flatcache::TokenC alpha_mode = carb::flatcache::asInt(pxr::UsdTokens->alpha_mode);
const carb::flatcache::TokenC base_alpha = carb::flatcache::asInt(pxr::UsdTokens->base_alpha);
const carb::flatcache::TokenC base_color_factor = carb::flatcache::asInt(pxr::UsdTokens->base_color_factor);
const carb::flatcache::TokenC base_color_texture = carb::flatcache::asInt(pxr::UsdTokens->base_color_texture);
const carb::flatcache::TokenC baseColorTex = carb::flatcache::asInt(pxr::UsdTokens->baseColorTex);
const carb::flatcache::TokenC constant = carb::flatcache::asInt(pxr::UsdTokens->constant);
const carb::flatcache::TokenC displacement = carb::flatcache::asInt(pxr::UsdTokens->displacement);
const carb::flatcache::TokenC doubleSided = carb::flatcache::asInt(pxr::UsdTokens->doubleSided);
const carb::flatcache::TokenC emissive_factor = carb::flatcache::asInt(pxr::UsdTokens->emissive_factor);
const carb::flatcache::TokenC faceVertexCounts = carb::flatcache::asInt(pxr::UsdTokens->faceVertexCounts);
const carb::flatcache::TokenC faceVertexIndices = carb::flatcache::asInt(pxr::UsdTokens->faceVertexIndices);
const carb::flatcache::TokenC gltf_material = carb::flatcache::asInt(pxr::UsdTokens->gltf_material);
const carb::flatcache::TokenC gltf_pbr_mdl = carb::flatcache::asInt(pxr::UsdTokens->gltf_pbr_mdl);
const carb::flatcache::TokenC gltf_texture_lookup = carb::flatcache::asInt(pxr::UsdTokens->gltf_texture_lookup);
const carb::flatcache::TokenC info_id = carb::flatcache::asInt(pxr::UsdTokens->info_id);
const carb::flatcache::TokenC info_sourceAsset_subIdentifier = carb::flatcache::asInt(pxr::UsdTokens->info_sourceAsset_subIdentifier);
const carb::flatcache::TokenC Material = carb::flatcache::asInt(pxr::UsdTokens->Material);
const carb::flatcache::TokenC materialId = carb::flatcache::asInt(pxr::UsdTokens->materialId);
const carb::flatcache::TokenC MaterialNetwork = carb::flatcache::asInt(pxr::UsdTokens->MaterialNetwork);
const carb::flatcache::TokenC Mesh = carb::flatcache::asInt(pxr::UsdTokens->Mesh);
const carb::flatcache::TokenC metallic_factor = carb::flatcache::asInt(pxr::UsdTokens->metallic_factor);
const carb::flatcache::TokenC none = carb::flatcache::asInt(pxr::UsdTokens->none);
const carb::flatcache::TokenC offset = carb::flatcache::asInt(pxr::UsdTokens->offset);
const carb::flatcache::TokenC out = carb::flatcache::asInt(pxr::UsdTokens->out);
const carb::flatcache::TokenC points = carb::flatcache::asInt(pxr::UsdTokens->points);
const carb::flatcache::TokenC primvarInterpolations = carb::flatcache::asInt(pxr::UsdTokens->primvarInterpolations);
const carb::flatcache::TokenC primvars = carb::flatcache::asInt(pxr::UsdTokens->primvars);
const carb::flatcache::TokenC primvars_displayColor = carb::flatcache::asInt(pxr::UsdTokens->primvars_displayColor);
const carb::flatcache::TokenC primvars_displayOpacity = carb::flatcache::asInt(pxr::UsdTokens->primvars_displayOpacity);
const carb::flatcache::TokenC primvars_normals = carb::flatcache::asInt(pxr::UsdTokens->primvars_normals);
const carb::flatcache::TokenC primvars_st = carb::flatcache::asInt(pxr::UsdTokens->primvars_st);
const carb::flatcache::TokenC primvars_vertexColor = carb::flatcache::asInt(pxr::UsdTokens->primvars_vertexColor);
const carb::flatcache::TokenC rotation = carb::flatcache::asInt(pxr::UsdTokens->rotation);
const carb::flatcache::TokenC roughness_factor = carb::flatcache::asInt(pxr::UsdTokens->roughness_factor);
const carb::flatcache::TokenC scale = carb::flatcache::asInt(pxr::UsdTokens->scale);
const carb::flatcache::TokenC Shader = carb::flatcache::asInt(pxr::UsdTokens->Shader);
const carb::flatcache::TokenC subdivisionScheme = carb::flatcache::asInt(pxr::UsdTokens->subdivisionScheme);
const carb::flatcache::TokenC surface = carb::flatcache::asInt(pxr::UsdTokens->surface);
const carb::flatcache::TokenC tex_coord_index = carb::flatcache::asInt(pxr::UsdTokens->tex_coord_index);
const carb::flatcache::TokenC texture = carb::flatcache::asInt(pxr::UsdTokens->texture);
const carb::flatcache::TokenC vertex = carb::flatcache::asInt(pxr::UsdTokens->vertex);
const carb::flatcache::TokenC vertex_color_name = carb::flatcache::asInt(pxr::UsdTokens->vertex_color_name);
const carb::flatcache::TokenC vertexColor = carb::flatcache::asInt(pxr::UsdTokens->vertexColor);
const carb::flatcache::TokenC wrap_s = carb::flatcache::asInt(pxr::UsdTokens->wrap_s);
const carb::flatcache::TokenC wrap_t = carb::flatcache::asInt(pxr::UsdTokens->wrap_t);
const carb::flatcache::TokenC _auto = carb::flatcache::asInt(pxr::UsdTokens->_auto);
const carb::flatcache::TokenC _cesium_localToEcefTransform = carb::flatcache::asInt(pxr::UsdTokens->_cesium_localToEcefTransform);
const carb::flatcache::TokenC _cesium_tileId = carb::flatcache::asInt(pxr::UsdTokens->_cesium_tileId);
const carb::flatcache::TokenC _cesium_tilesetId = carb::flatcache::asInt(pxr::UsdTokens->_cesium_tilesetId);
const carb::flatcache::TokenC _deletedPrims = carb::flatcache::asInt(pxr::UsdTokens->_deletedPrims);
const carb::flatcache::TokenC _localExtent = carb::flatcache::asInt(pxr::UsdTokens->_localExtent);
const carb::flatcache::TokenC _nodePaths = carb::flatcache::asInt(pxr::UsdTokens->_nodePaths);
const carb::flatcache::TokenC _paramColorSpace = carb::flatcache::asInt(pxr::UsdTokens->_paramColorSpace);
const carb::flatcache::TokenC _parameters = carb::flatcache::asInt(pxr::UsdTokens->_parameters);
const carb::flatcache::TokenC _relationships_inputId = carb::flatcache::asInt(pxr::UsdTokens->_relationships_inputId);
const carb::flatcache::TokenC _relationships_inputName = carb::flatcache::asInt(pxr::UsdTokens->_relationships_inputName);
const carb::flatcache::TokenC _relationships_outputId = carb::flatcache::asInt(pxr::UsdTokens->_relationships_outputId);
const carb::flatcache::TokenC _relationships_outputName = carb::flatcache::asInt(pxr::UsdTokens->_relationships_outputName);
const carb::flatcache::TokenC _terminals = carb::flatcache::asInt(pxr::UsdTokens->_terminals);
const carb::flatcache::TokenC _worldExtent = carb::flatcache::asInt(pxr::UsdTokens->_worldExtent);
const carb::flatcache::TokenC _worldOrientation = carb::flatcache::asInt(pxr::UsdTokens->_worldOrientation);
const carb::flatcache::TokenC _worldPosition = carb::flatcache::asInt(pxr::UsdTokens->_worldPosition);
const carb::flatcache::TokenC _worldScale = carb::flatcache::asInt(pxr::UsdTokens->_worldScale);
const carb::flatcache::TokenC _worldVisibility = carb::flatcache::asInt(pxr::UsdTokens->_worldVisibility);
}

namespace cesium::omniverse::UsdTokens {
const pxr::TfToken& alpha_cutoff = pxr::UsdTokens->alpha_cutoff;
const pxr::TfToken& alpha_mode = pxr::UsdTokens->alpha_mode;
const pxr::TfToken& base_alpha = pxr::UsdTokens->base_alpha;
const pxr::TfToken& base_color_factor = pxr::UsdTokens->base_color_factor;
const pxr::TfToken& base_color_texture = pxr::UsdTokens->base_color_texture;
const pxr::TfToken& baseColorTex = pxr::UsdTokens->baseColorTex;
const pxr::TfToken& constant = pxr::UsdTokens->constant;
const pxr::TfToken& displacement = pxr::UsdTokens->displacement;
const pxr::TfToken& doubleSided = pxr::UsdTokens->doubleSided;
const pxr::TfToken& emissive_factor = pxr::UsdTokens->emissive_factor;
const pxr::TfToken& faceVertexCounts = pxr::UsdTokens->faceVertexCounts;
const pxr::TfToken& faceVertexIndices = pxr::UsdTokens->faceVertexIndices;
const pxr::TfToken& gltf_material = pxr::UsdTokens->gltf_material;
const pxr::TfToken& gltf_pbr_mdl = pxr::UsdTokens->gltf_pbr_mdl;
const pxr::TfToken& gltf_texture_lookup = pxr::UsdTokens->gltf_texture_lookup;
const pxr::TfToken& info_id = pxr::UsdTokens->info_id;
const pxr::TfToken& info_sourceAsset_subIdentifier = pxr::UsdTokens->info_sourceAsset_subIdentifier;
const pxr::TfToken& Material = pxr::UsdTokens->Material;
const pxr::TfToken& materialId = pxr::UsdTokens->materialId;
const pxr::TfToken& MaterialNetwork = pxr::UsdTokens->MaterialNetwork;
const pxr::TfToken& Mesh = pxr::UsdTokens->Mesh;
const pxr::TfToken& metallic_factor = pxr::UsdTokens->metallic_factor;
const pxr::TfToken& none = pxr::UsdTokens->none;
const pxr::TfToken& offset = pxr::UsdTokens->offset;
const pxr::TfToken& out = pxr::UsdTokens->out;
const pxr::TfToken& points = pxr::UsdTokens->points;
const pxr::TfToken& primvarInterpolations = pxr::UsdTokens->primvarInterpolations;
const pxr::TfToken& primvars = pxr::UsdTokens->primvars;
const pxr::TfToken& primvars_displayColor = pxr::UsdTokens->primvars_displayColor;
const pxr::TfToken& primvars_displayOpacity = pxr::UsdTokens->primvars_displayOpacity;
const pxr::TfToken& primvars_normals = pxr::UsdTokens->primvars_normals;
const pxr::TfToken& primvars_st = pxr::UsdTokens->primvars_st;
const pxr::TfToken& primvars_vertexColor = pxr::UsdTokens->primvars_vertexColor;
const pxr::TfToken& rotation = pxr::UsdTokens->rotation;
const pxr::TfToken& roughness_factor = pxr::UsdTokens->roughness_factor;
const pxr::TfToken& scale = pxr::UsdTokens->scale;
const pxr::TfToken& Shader = pxr::UsdTokens->Shader;
const pxr::TfToken& subdivisionScheme = pxr::UsdTokens->subdivisionScheme;
const pxr::TfToken& surface = pxr::UsdTokens->surface;
const pxr::TfToken& tex_coord_index = pxr::UsdTokens->tex_coord_index;
const pxr::TfToken& texture = pxr::UsdTokens->texture;
const pxr::TfToken& vertex = pxr::UsdTokens->vertex;
const pxr::TfToken& vertex_color_name = pxr::UsdTokens->vertex_color_name;
const pxr::TfToken& vertexColor = pxr::UsdTokens->vertexColor;
const pxr::TfToken& wrap_s = pxr::UsdTokens->wrap_s;
const pxr::TfToken& wrap_t = pxr::UsdTokens->wrap_t;
const pxr::TfToken& _auto = pxr::UsdTokens->_auto;
const pxr::TfToken& _cesium_localToEcefTransform = pxr::UsdTokens->_cesium_localToEcefTransform;
const pxr::TfToken& _cesium_tileId = pxr::UsdTokens->_cesium_tileId;
const pxr::TfToken& _cesium_tilesetId = pxr::UsdTokens->_cesium_tilesetId;
const pxr::TfToken& _deletedPrims = pxr::UsdTokens->_deletedPrims;
const pxr::TfToken& _localExtent = pxr::UsdTokens->_localExtent;
const pxr::TfToken& _nodePaths = pxr::UsdTokens->_nodePaths;
const pxr::TfToken& _paramColorSpace = pxr::UsdTokens->_paramColorSpace;
const pxr::TfToken& _parameters = pxr::UsdTokens->_parameters;
const pxr::TfToken& _relationships_inputId = pxr::UsdTokens->_relationships_inputId;
const pxr::TfToken& _relationships_inputName = pxr::UsdTokens->_relationships_inputName;
const pxr::TfToken& _relationships_outputId = pxr::UsdTokens->_relationships_outputId;
const pxr::TfToken& _relationships_outputName = pxr::UsdTokens->_relationships_outputName;
const pxr::TfToken& _terminals = pxr::UsdTokens->_terminals;
const pxr::TfToken& _worldExtent = pxr::UsdTokens->_worldExtent;
const pxr::TfToken& _worldOrientation = pxr::UsdTokens->_worldOrientation;
const pxr::TfToken& _worldPosition = pxr::UsdTokens->_worldPosition;
const pxr::TfToken& _worldScale = pxr::UsdTokens->_worldScale;
const pxr::TfToken& _worldVisibility = pxr::UsdTokens->_worldVisibility;
}
// clang-format on
