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
    (a)
    (add)
    (albedo_add)
    (b)
    (constant)
    (diffuse_texture)
    (displacement)
    (doubleSided)
    (faceVertexCounts)
    (faceVertexIndices)
    (id)
    (Material)
    (Mesh)
    (multiply)
    (none)
    (OmniPBR)
    (out)
    (points)
    (primvarInterpolations)
    (primvars)
    (Shader)
    (sourceAsset)
    (subdivisionScheme)
    (surface)
    (texture_coordinate_2d)
    (vertex)
    (visibility)
    (_cesium_localToEcefTransform)
    (_cesium_tileId)
    (_cesium_tilesetId)
    (_deletedPrims)
    (_localExtent)
    (_localMatrix)
    (_nodePaths)
    (_paramColorSpace)
    (_relationship_ids)
    (_relationship_names)
    (_sdrMetadata)
    (_terminal_names)
    (_terminal_sourceIds)
    (_terminal_sourceNames)
    (_worldExtent)
    (_worldOrientation)
    (_worldPosition)
    (_worldScale)
    (_worldVisibility)
    ((_auto, "auto"))
    ((add_float2_float2, "add(float2,float2)"))
    ((coord, "inputs:coord"))
    ((diffuse_color_constant, "inputs:diffuse_color_constant"))
    ((info_id, "info:id"))
    ((info_implementationSource, "info:implementationSource"))
    ((info_mdl_sourceAsset, "info:mdl:sourceAsset"))
    ((info_mdl_sourceAsset_subIdentifier, "info:mdl:sourceAsset:subIdentifier"))
    ((info_sourceAsset_subIdentifier, "info:sourceAsset:subIdentifier"))
    ((lookup_color, "lookup_color"))
    ((materialBinding, "material:binding"))
    ((metallic_constant, "inputs:metallic_constant"))
    ((multiply_float2_float2, "multiply(float2,float2)"))
    ((nvidia_support_definitions_mdl, "nvidia/support_definitions.mdl"))
    ((OmniPBR_mdl, "OmniPBR.mdl"))
    ((outputs_displacement, "outputs:displacement"))
    ((outputs_out, "outputs:out"))
    ((outputs_surface, "outputs:surface"))
    ((primvars_displayColor, "primvars:displayColor"))
    ((primvars_normals, "primvars:normals"))
    ((primvars_st, "primvars:st"))
    ((reflection_roughness_constant, "inputs:reflection_roughness_constant"))
    ((specular_level, "inputs:specular_level"))
    ((tex, "inputs:tex"))
    ((wrap_u, "inputs:wrap_u"))
    ((wrap_v, "inputs:wrap_v"))
);

#ifdef CESIUM_OMNI_MSVC
__pragma(warning(pop))
#endif

}

namespace cesium::omniverse::FabricTokens {
const omni::fabric::TokenC a = omni::fabric::asInt(pxr::UsdTokens->a);
const omni::fabric::TokenC add = omni::fabric::asInt(pxr::UsdTokens->add);
const omni::fabric::TokenC add_float2_float2 = omni::fabric::asInt(pxr::UsdTokens->add_float2_float2);
const omni::fabric::TokenC albedo_add = omni::fabric::asInt(pxr::UsdTokens->albedo_add);
const omni::fabric::TokenC b = omni::fabric::asInt(pxr::UsdTokens->b);
const omni::fabric::TokenC constant = omni::fabric::asInt(pxr::UsdTokens->constant);
const omni::fabric::TokenC coord = omni::fabric::asInt(pxr::UsdTokens->coord);
const omni::fabric::TokenC diffuse_color_constant = omni::fabric::asInt(pxr::UsdTokens->diffuse_color_constant);
const omni::fabric::TokenC diffuse_texture = omni::fabric::asInt(pxr::UsdTokens->diffuse_texture);
const omni::fabric::TokenC displacement = omni::fabric::asInt(pxr::UsdTokens->displacement);
const omni::fabric::TokenC doubleSided = omni::fabric::asInt(pxr::UsdTokens->doubleSided);
const omni::fabric::TokenC faceVertexCounts = omni::fabric::asInt(pxr::UsdTokens->faceVertexCounts);
const omni::fabric::TokenC faceVertexIndices = omni::fabric::asInt(pxr::UsdTokens->faceVertexIndices);
const omni::fabric::TokenC id = omni::fabric::asInt(pxr::UsdTokens->id);
const omni::fabric::TokenC info_id = omni::fabric::asInt(pxr::UsdTokens->info_id);
const omni::fabric::TokenC info_implementationSource = omni::fabric::asInt(pxr::UsdTokens->info_implementationSource);
const omni::fabric::TokenC info_mdl_sourceAsset = omni::fabric::asInt(pxr::UsdTokens->info_mdl_sourceAsset);
const omni::fabric::TokenC info_mdl_sourceAsset_subIdentifier = omni::fabric::asInt(pxr::UsdTokens->info_mdl_sourceAsset_subIdentifier);
const omni::fabric::TokenC info_sourceAsset_subIdentifier = omni::fabric::asInt(pxr::UsdTokens->info_sourceAsset_subIdentifier);
const omni::fabric::TokenC lookup_color = omni::fabric::asInt(pxr::UsdTokens->lookup_color);
const omni::fabric::TokenC Material = omni::fabric::asInt(pxr::UsdTokens->Material);
const omni::fabric::TokenC materialBinding = omni::fabric::asInt(pxr::UsdTokens->materialBinding);
const omni::fabric::TokenC Mesh = omni::fabric::asInt(pxr::UsdTokens->Mesh);
const omni::fabric::TokenC metallic_constant = omni::fabric::asInt(pxr::UsdTokens->metallic_constant);
const omni::fabric::TokenC multiply = omni::fabric::asInt(pxr::UsdTokens->multiply);
const omni::fabric::TokenC multiply_float2_float2 = omni::fabric::asInt(pxr::UsdTokens->multiply_float2_float2);
const omni::fabric::TokenC none = omni::fabric::asInt(pxr::UsdTokens->none);
const omni::fabric::TokenC nvidia_support_definitions_mdl = omni::fabric::asInt(pxr::UsdTokens->nvidia_support_definitions_mdl);
const omni::fabric::TokenC OmniPBR = omni::fabric::asInt(pxr::UsdTokens->OmniPBR);
const omni::fabric::TokenC OmniPBR_mdl = omni::fabric::asInt(pxr::UsdTokens->OmniPBR_mdl);
const omni::fabric::TokenC out = omni::fabric::asInt(pxr::UsdTokens->out);
const omni::fabric::TokenC outputs_displacement = omni::fabric::asInt(pxr::UsdTokens->outputs_displacement);
const omni::fabric::TokenC outputs_out = omni::fabric::asInt(pxr::UsdTokens->outputs_out);
const omni::fabric::TokenC outputs_surface = omni::fabric::asInt(pxr::UsdTokens->outputs_surface);
const omni::fabric::TokenC points = omni::fabric::asInt(pxr::UsdTokens->points);
const omni::fabric::TokenC primvarInterpolations = omni::fabric::asInt(pxr::UsdTokens->primvarInterpolations);
const omni::fabric::TokenC primvars = omni::fabric::asInt(pxr::UsdTokens->primvars);
const omni::fabric::TokenC primvars_displayColor = omni::fabric::asInt(pxr::UsdTokens->primvars_displayColor);
const omni::fabric::TokenC primvars_normals = omni::fabric::asInt(pxr::UsdTokens->primvars_normals);
const omni::fabric::TokenC primvars_st = omni::fabric::asInt(pxr::UsdTokens->primvars_st);
const omni::fabric::TokenC reflection_roughness_constant = omni::fabric::asInt(pxr::UsdTokens->reflection_roughness_constant);
const omni::fabric::TokenC Shader = omni::fabric::asInt(pxr::UsdTokens->Shader);
const omni::fabric::TokenC sourceAsset = omni::fabric::asInt(pxr::UsdTokens->sourceAsset);
const omni::fabric::TokenC specular_level = omni::fabric::asInt(pxr::UsdTokens->specular_level);
const omni::fabric::TokenC subdivisionScheme = omni::fabric::asInt(pxr::UsdTokens->subdivisionScheme);
const omni::fabric::TokenC surface = omni::fabric::asInt(pxr::UsdTokens->surface);
const omni::fabric::TokenC tex = omni::fabric::asInt(pxr::UsdTokens->tex);
const omni::fabric::TokenC texture_coordinate_2d = omni::fabric::asInt(pxr::UsdTokens->texture_coordinate_2d);
const omni::fabric::TokenC vertex = omni::fabric::asInt(pxr::UsdTokens->vertex);
const omni::fabric::TokenC visibility = omni::fabric::asInt(pxr::UsdTokens->visibility);
const omni::fabric::TokenC wrap_u = omni::fabric::asInt(pxr::UsdTokens->wrap_u);
const omni::fabric::TokenC wrap_v = omni::fabric::asInt(pxr::UsdTokens->wrap_v);
const omni::fabric::TokenC _auto = omni::fabric::asInt(pxr::UsdTokens->_auto);
const omni::fabric::TokenC _cesium_localToEcefTransform = omni::fabric::asInt(pxr::UsdTokens->_cesium_localToEcefTransform);
const omni::fabric::TokenC _cesium_tileId = omni::fabric::asInt(pxr::UsdTokens->_cesium_tileId);
const omni::fabric::TokenC _cesium_tilesetId = omni::fabric::asInt(pxr::UsdTokens->_cesium_tilesetId);
const omni::fabric::TokenC _deletedPrims = omni::fabric::asInt(pxr::UsdTokens->_deletedPrims);
const omni::fabric::TokenC _localExtent = omni::fabric::asInt(pxr::UsdTokens->_localExtent);
const omni::fabric::TokenC _localMatrix = omni::fabric::asInt(pxr::UsdTokens->_localMatrix);
const omni::fabric::TokenC _nodePaths = omni::fabric::asInt(pxr::UsdTokens->_nodePaths);
const omni::fabric::TokenC _paramColorSpace = omni::fabric::asInt(pxr::UsdTokens->_paramColorSpace);
const omni::fabric::TokenC _relationship_ids = omni::fabric::asInt(pxr::UsdTokens->_relationship_ids);
const omni::fabric::TokenC _relationship_names = omni::fabric::asInt(pxr::UsdTokens->_relationship_names);
const omni::fabric::TokenC _sdrMetadata = omni::fabric::asInt(pxr::UsdTokens->_sdrMetadata);
const omni::fabric::TokenC _terminal_names = omni::fabric::asInt(pxr::UsdTokens->_terminal_names);
const omni::fabric::TokenC _terminal_sourceIds = omni::fabric::asInt(pxr::UsdTokens->_terminal_sourceIds);
const omni::fabric::TokenC _terminal_sourceNames = omni::fabric::asInt(pxr::UsdTokens->_terminal_sourceNames);
const omni::fabric::TokenC _worldExtent = omni::fabric::asInt(pxr::UsdTokens->_worldExtent);
const omni::fabric::TokenC _worldOrientation = omni::fabric::asInt(pxr::UsdTokens->_worldOrientation);
const omni::fabric::TokenC _worldPosition = omni::fabric::asInt(pxr::UsdTokens->_worldPosition);
const omni::fabric::TokenC _worldScale = omni::fabric::asInt(pxr::UsdTokens->_worldScale);
const omni::fabric::TokenC _worldVisibility = omni::fabric::asInt(pxr::UsdTokens->_worldVisibility);
}

namespace cesium::omniverse::UsdTokens {
const pxr::TfToken& a = pxr::UsdTokens->a;
const pxr::TfToken& add = pxr::UsdTokens->add;
const pxr::TfToken& add_float2_float2 = pxr::UsdTokens->add_float2_float2;
const pxr::TfToken& albedo_add = pxr::UsdTokens->albedo_add;
const pxr::TfToken& b = pxr::UsdTokens->b;
const pxr::TfToken& constant = pxr::UsdTokens->constant;
const pxr::TfToken& coord = pxr::UsdTokens->coord;
const pxr::TfToken& diffuse_color_constant = pxr::UsdTokens->diffuse_color_constant;
const pxr::TfToken& diffuse_texture = pxr::UsdTokens->diffuse_texture;
const pxr::TfToken& displacement = pxr::UsdTokens->displacement;
const pxr::TfToken& doubleSided = pxr::UsdTokens->doubleSided;
const pxr::TfToken& faceVertexCounts = pxr::UsdTokens->faceVertexCounts;
const pxr::TfToken& faceVertexIndices = pxr::UsdTokens->faceVertexIndices;
const pxr::TfToken& id = pxr::UsdTokens->id;
const pxr::TfToken& info_id = pxr::UsdTokens->info_id;
const pxr::TfToken& info_implementationSource = pxr::UsdTokens->info_implementationSource;
const pxr::TfToken& info_mdl_sourceAsset = pxr::UsdTokens->info_mdl_sourceAsset;
const pxr::TfToken& info_mdl_sourceAsset_subIdentifier = pxr::UsdTokens->info_mdl_sourceAsset_subIdentifier;
const pxr::TfToken& info_sourceAsset_subIdentifier = pxr::UsdTokens->info_sourceAsset_subIdentifier;
const pxr::TfToken& lookup_color = pxr::UsdTokens->lookup_color;
const pxr::TfToken& Material = pxr::UsdTokens->Material;
const pxr::TfToken& materialBinding = pxr::UsdTokens->materialBinding;
const pxr::TfToken& Mesh = pxr::UsdTokens->Mesh;
const pxr::TfToken& metallic_constant = pxr::UsdTokens->metallic_constant;
const pxr::TfToken& multiply = pxr::UsdTokens->multiply;
const pxr::TfToken& multiply_float2_float2 = pxr::UsdTokens->multiply_float2_float2;
const pxr::TfToken& none = pxr::UsdTokens->none;
const pxr::TfToken& nvidia_support_definitions_mdl = pxr::UsdTokens->nvidia_support_definitions_mdl;
const pxr::TfToken& OmniPBR = pxr::UsdTokens->OmniPBR;
const pxr::TfToken& OmniPBR_mdl = pxr::UsdTokens->OmniPBR_mdl;
const pxr::TfToken& out = pxr::UsdTokens->out;
const pxr::TfToken& outputs_displacement = pxr::UsdTokens->outputs_displacement;
const pxr::TfToken& outputs_out = pxr::UsdTokens->outputs_out;
const pxr::TfToken& outputs_surface = pxr::UsdTokens->outputs_surface;
const pxr::TfToken& points = pxr::UsdTokens->points;
const pxr::TfToken& primvarInterpolations = pxr::UsdTokens->primvarInterpolations;
const pxr::TfToken& primvars = pxr::UsdTokens->primvars;
const pxr::TfToken& primvars_displayColor = pxr::UsdTokens->primvars_displayColor;
const pxr::TfToken& primvars_normals = pxr::UsdTokens->primvars_normals;
const pxr::TfToken& primvars_st = pxr::UsdTokens->primvars_st;
const pxr::TfToken& reflection_roughness_constant = pxr::UsdTokens->reflection_roughness_constant;
const pxr::TfToken& Shader = pxr::UsdTokens->Shader;
const pxr::TfToken& sourceAsset = pxr::UsdTokens->sourceAsset;
const pxr::TfToken& specular_level = pxr::UsdTokens->specular_level;
const pxr::TfToken& subdivisionScheme = pxr::UsdTokens->subdivisionScheme;
const pxr::TfToken& surface = pxr::UsdTokens->surface;
const pxr::TfToken& tex = pxr::UsdTokens->tex;
const pxr::TfToken& texture_coordinate_2d = pxr::UsdTokens->texture_coordinate_2d;
const pxr::TfToken& vertex = pxr::UsdTokens->vertex;
const pxr::TfToken& visibility = pxr::UsdTokens->visibility;
const pxr::TfToken& wrap_u = pxr::UsdTokens->wrap_u;
const pxr::TfToken& wrap_v = pxr::UsdTokens->wrap_v;
const pxr::TfToken& _auto = pxr::UsdTokens->_auto;
const pxr::TfToken& _cesium_localToEcefTransform = pxr::UsdTokens->_cesium_localToEcefTransform;
const pxr::TfToken& _cesium_tileId = pxr::UsdTokens->_cesium_tileId;
const pxr::TfToken& _cesium_tilesetId = pxr::UsdTokens->_cesium_tilesetId;
const pxr::TfToken& _deletedPrims = pxr::UsdTokens->_deletedPrims;
const pxr::TfToken& _localExtent = pxr::UsdTokens->_localExtent;
const pxr::TfToken& _localMatrix = pxr::UsdTokens->_localMatrix;
const pxr::TfToken& _nodePaths = pxr::UsdTokens->_nodePaths;
const pxr::TfToken& _paramColorSpace = pxr::UsdTokens->_paramColorSpace;
const pxr::TfToken& _relationship_ids = pxr::UsdTokens->_relationship_ids;
const pxr::TfToken& _relationship_names = pxr::UsdTokens->_relationship_names;
const pxr::TfToken& _sdrMetadata = pxr::UsdTokens->_sdrMetadata;
const pxr::TfToken& _terminal_names = pxr::UsdTokens->_terminal_names;
const pxr::TfToken& _terminal_sourceIds = pxr::UsdTokens->_terminal_sourceIds;
const pxr::TfToken& _terminal_sourceNames = pxr::UsdTokens->_terminal_sourceNames;
const pxr::TfToken& _worldExtent = pxr::UsdTokens->_worldExtent;
const pxr::TfToken& _worldOrientation = pxr::UsdTokens->_worldOrientation;
const pxr::TfToken& _worldPosition = pxr::UsdTokens->_worldPosition;
const pxr::TfToken& _worldScale = pxr::UsdTokens->_worldScale;
const pxr::TfToken& _worldVisibility = pxr::UsdTokens->_worldVisibility;
}
// clang-format on
