package faces.lib

/*
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * This code was adapted from
 * https://github.com/unibas-gravis/scalismo-faces
 * Copyright University of Basel, Graphics and Vision Research Group
 */


import scalismo.color.RGBA
import scalismo.geometry.{Point, _3D}
import scalismo.mesh.{SurfacePointProperty, TriangleMesh3D, VertexColorMesh3D}

/**
 * colored mesh with RGBA specular and diffuse Albedo per vertex
 * @param shape positions
 * @param diffuseAlbedo diffuse albedo of mesh surface, per point
 * @param specularAlbedo specular of mesh surface, per point
 */
case class VertexAlbedoMesh3D(shape: TriangleMesh3D, diffuseAlbedo: SurfacePointProperty[RGBA], specularAlbedo: SurfacePointProperty[RGBA]) {
  require(shape.triangulation == diffuseAlbedo.triangulation)
  require(shape.triangulation == specularAlbedo.triangulation)

  def transform(trafo: Point[_3D] => Point[_3D]): VertexAlbedoMesh3D = {
    val s = shape.transform { trafo }
    copy(shape = s)
  }

  def diffuseAlbedoMesh(): VertexColorMesh3D ={
    VertexColorMesh3D(shape, diffuseAlbedo)
  }

  def specularAlbedoMesh(): VertexColorMesh3D ={
    VertexColorMesh3D(shape, specularAlbedo)
  }
}