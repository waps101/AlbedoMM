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


import java.io.File
import java.net.URI

import breeze.linalg.DenseVector
import breeze.linalg.DenseVector
import breeze.numerics.pow
import scalismo.color.RGBA
import scalismo.faces.parameters.{Camera, ColorTransform, DirectionalLight, Illumination, ImageSize, MoMoInstance, ParametricRenderer, Pose, RenderObject, RenderParameter, SphericalHarmonicsLight, ViewParameter}
import scalismo.faces.sampling.face.{ParametricImageRenderer, ParametricLandmarksRenderer, ParametricMaskRenderer, ParametricMeshRenderer, ParametricModel}
import scalismo.geometry.{EuclideanVector, EuclideanVector3D, Point, Point2D, Point3D, _3D}
import scalismo.mesh.{BarycentricCoordinates, MeshSurfaceProperty, SurfacePointProperty, TriangleId, TriangleMesh3D, VertexColorMesh3D}
import scalismo.utils.Memoize
import scalismo.faces.image.{PixelImage, PixelImageOperations}
import scalismo.faces.render.{Affine3D, PixelShader, PixelShaders, PointShader, TriangleFilters, TriangleRenderer, ZBuffer}
import scalismo.faces.io.{MeshIO, MoMoIO, PixelImageIO, RenderParameterIO}
import scalismo.faces.landmarks.TLMSLandmark2D
import scalismo.faces.mesh.ColorNormalMesh3D
import scalismo.faces.numerics.SphericalHarmonics
import scalismo.faces.render.PixelShaders.{SphericalHarmonicsLambertShader, SphericalHarmonicsSpecularShader}

import scala.reflect.ClassTag

object SpecularRenderingTest extends App{
  scalismo.initialize()

  val model = AlbedoMoMoIO.read(new File("/media/data/work/albedomodel/pipeline-data/data/modelbuilding/model/yamm2020_bfm_nomouth_combinedModel.h5")).get

  val renderer = AlbedoMoMoRenderer(model)

  val bip = new File("/media/data/work/Ilker/reducedBIP/data/parameters/").listFiles().take(10)

  bip.zipWithIndex.foreach(i =>{

    val ill = RenderParameterIO.read(i._1).get
    val img = renderer.renderImage(RenderParameter.default.withMoMo(MoMoInstance(IndexedSeq(0.0), IndexedSeq(0.0), IndexedSeq(0.0), new URI(""))).fitToImageSize(1000,1000).withEnvironmentMap(ill.environmentMap))
    PixelImageIO.write(img, new File("/media/temp/speculartest/test_"+i._2+".png"))
  })
  val img = renderer.renderImage(RenderParameter.default.withMoMo(MoMoInstance(IndexedSeq(0.0), IndexedSeq(0.0), IndexedSeq(0.0), new URI(""))).fitToImageSize(1000,1000))
  PixelImageIO.write(img, new File("/media/temp/speculartest/test.png"))
}


/** parametric renderer for a Morphable Model, implements all useful Parameteric*Renderer interfaces */
class AlbedoMoMoRenderer(val model: AlbedoMoMo, val specularExponent: Double, val clearColor: RGBA)
  extends ParametricImageRenderer[RGBA]
    with ParametricLandmarksRenderer
    with ParametricMaskRenderer
    with ParametricMeshRenderer
    with ParametricAlbedoModel {

  /** pad a coefficient vector if it is too short, basis with single vector */
  private def padCoefficients(coefficients: DenseVector[Double], rank: Int): DenseVector[Double] = {
    require(coefficients.length <= rank, "too many coefficients for model")
    if (coefficients.length == rank)
      coefficients
    else
      DenseVector(coefficients.toArray ++ Array.fill(rank - coefficients.length)(0.0))
  }

  /** create an instance of the model, in the original model's object coordinates */
  override def instance(parameters: RenderParameter): VertexAlbedoMesh3D = {
    model.instance(parameters.momo.coefficients)
  }


  /** render the image described by the parameters */
  override def renderImage(parameters: RenderParameter): PixelImage[RGBA] = {
    val inst = instance(parameters)
    val imgSpec =renderImageSpecular(parameters, inst)

    val imgDiffuse =  renderImageDiffuse(parameters, inst)

  imgDiffuse.zip(imgSpec).map(p => (p._1+p._2).clamped)
  }

  def renderImageDiffuse(parameters: RenderParameter): PixelImage[RGBA] = {
    val inst = instance(parameters)
    ParametricRenderer.renderParameterVertexColorMesh(
      parameters,
      inst.diffuseAlbedoMesh(),
      clearColor)
  }

  def renderImageSpecular(parameters: RenderParameter): PixelImage[RGBA] = {
    val inst = instance(parameters)
    SpecularParametricRenderer.specularRenderParameterVertexColorMesh(
      parameters,
      inst,
      specularExponent,
      clearColor)
  }

  def renderImageDiffuse(parameters: RenderParameter, inst: VertexAlbedoMesh3D): PixelImage[RGBA] = {
    ParametricRenderer.renderParameterVertexColorMesh(
      parameters,
      inst.diffuseAlbedoMesh(),
      clearColor)
  }

  def renderImageSpecular(parameters: RenderParameter, inst: VertexAlbedoMesh3D): PixelImage[RGBA] = {
    SpecularParametricRenderer.specularRenderParameterVertexColorMesh(
      parameters,
      inst,
      specularExponent,
      clearColor)
  }


  /** render the mesh described by the parameters, draws instance from model and places properly in the world (world coordinates) */
  override def renderMesh(parameters: RenderParameter): VertexColorMesh3D = {
    val t = parameters.pose.transform
    val mesh = instance(parameters)
    VertexColorMesh3D(
      mesh.shape.transform(p => t(p)),
      mesh.diffuseAlbedo
    )
  }

  /** render landmark position in the image */
  override def renderLandmark(lmId: String, parameter: RenderParameter): Option[TLMSLandmark2D] = {
    val renderer = parameter.renderTransform
    for {
      ptId <- model.landmarkPointId(lmId)
      lm3d <- Some(model.instanceAtPoint(parameter.momo.coefficients, ptId)._1)
      lmImage <- Some(renderer(lm3d))
    } yield TLMSLandmark2D(lmId, Point(lmImage.x, lmImage.y), visible = true)
  }

  /** checks the availability of a named landmark */
  override def hasLandmarkId(lmId: String): Boolean = model.landmarkPointId(lmId).isDefined


  /** get all available landmarks */
  override def allLandmarkIds: IndexedSeq[String] = model.landmarks.keySet.toIndexedSeq


  /** render a mask defined on the model to image space */
  override def renderMask(parameters: RenderParameter, mask: MeshSurfaceProperty[Int]): PixelImage[Int] = {
    val inst = instance(parameters)
    val maskImage = SpecularParametricRenderer.renderPropertyImage(parameters, inst.shape, mask)
    maskImage.map(_.getOrElse(0)) // 0 - invalid, outside rendering area
  }

  /** get a cached version of this renderer */
  def cached(cacheSize: Int) = new AlbedoMoMoRenderer(model,  specularExponent, clearColor) {
    private val imageRenderer = Memoize(super.renderImage, cacheSize)
    private val meshRenderer = Memoize(super.renderMesh, cacheSize)
    private val maskRenderer = Memoize((super.renderMask _).tupled, cacheSize)
    private val lmRenderer = Memoize((super.renderLandmark _).tupled, cacheSize * allLandmarkIds.length)
    private val instancer = Memoize(super.instance, cacheSize)

    override def renderImage(parameters: RenderParameter): PixelImage[RGBA] = imageRenderer(parameters)
    override def renderLandmark(lmId: String, parameter: RenderParameter): Option[TLMSLandmark2D] = lmRenderer((lmId, parameter))
    override def renderMesh(parameters: RenderParameter): VertexColorMesh3D = meshRenderer(parameters)
    override def instance(parameters: RenderParameter): VertexAlbedoMesh3D = instancer(parameters)
    override def renderMask(parameters: RenderParameter, mask: MeshSurfaceProperty[Int]): PixelImage[Int] = maskRenderer((parameters, mask))
  }
}

object AlbedoMoMoRenderer {
  def apply(model: AlbedoMoMo,  specularExponent: Double, clearColor: RGBA) = new AlbedoMoMoRenderer(model, specularExponent, clearColor)
  def apply(model: AlbedoMoMo,   specularExponent: Double) = new AlbedoMoMoRenderer(model, specularExponent, RGBA.BlackTransparent)
  def apply(model: AlbedoMoMo) = new AlbedoMoMoRenderer(model, 20.0,  RGBA.BlackTransparent)

}





object SpecularParametricRenderer {
  /**
   * render a mesh with specified colors and normals according to scene description parameter
   *
   * @param parameter  scene description
   * @param mesh       mesh to render, has positions, colors and normals
   * @param clearColor background color of buffer
   * @return
   */
  def specularRenderParameterMesh(parameter: RenderParameter,
                                  mesh: ColorNormalMesh3D,
                                  specular: SurfacePointProperty[RGBA],
                                  specularExp: Double,
                                  clearColor: RGBA = RGBA.BlackTransparent): PixelImage[RGBA] = {
    val buffer = ZBuffer(parameter.imageSize.width, parameter.imageSize.height, clearColor)

    val worldMesh = mesh.transform(parameter.modelViewTransform)
    val backfaceCullingFilter = TriangleFilters.backfaceCullingFilter(worldMesh.shape, parameter.view.eyePosition)

    def pixelShader(mesh: ColorNormalMesh3D,
                    specular: SurfacePointProperty[RGBA],
                    specularExp: Double
                   ): PixelShader[RGBA] = {
      val worldMesh = mesh.transform(parameter.pose.transform)
      val ctRGB = parameter.colorTransform.transform
      val ct = (c: RGBA) => ctRGB(c.toRGB).toRGBA
      /*   // default: both illumination sources active
         (environmentMap.shader(worldMesh, view.eyePosition) + directionalLight.shader(worldMesh, view.eyePosition)).map(ct)
         // old compatibility
         if (environmentMap.nonEmpty)*/
      val environmentMap = SpecularSphericalHarmonicsLight(parameter.environmentMap.coefficients)
      environmentMap.shader(worldMesh, parameter.view.eyePosition, specular, specularExp).map(ct)
      /*  else
          directionalLight.shader(worldMesh, view.eyePosition).map(ct)*/
    }

    TriangleRenderer.renderMesh(
      mesh.shape,
      backfaceCullingFilter,
      parameter.pointShader,
      parameter.imageSize.screenTransform,
      pixelShader(mesh, specular, specularExp),
      buffer).toImage
  }

  /**
   * render according to parameters, convenience for vertex color mesh with vertex normals
   *
   * @param parameter  scene description
   * @param mesh       mesh to render, vertex color, vertex normals
   * @param clearColor background color of buffer
   * @return
   */
  def specularRenderParameterVertexColorMesh(parameter: RenderParameter,
                                             mesh: VertexAlbedoMesh3D,
                                             specularExp: Double,
                                             clearColor: RGBA = RGBA.BlackTransparent): PixelImage[RGBA] = {
    specularRenderParameterMesh(parameter, ColorNormalMesh3D(VertexColorMesh3D(mesh.shape, mesh.diffuseAlbedo)), mesh.specularAlbedo, specularExp, clearColor)
  }


  /**
   * render an arbitrary property on the mesh into buffer (rasterization)
   *
   * @param renderParameter scene description
   * @param mesh            mesh to render, positions
   * @param property        surface property to rasterize
   * @tparam A type of surface property
   * @return
   */
  def renderPropertyImage[A: ClassTag](renderParameter: RenderParameter,
                                       mesh: TriangleMesh3D,
                                       property: MeshSurfaceProperty[A]): PixelImage[Option[A]] = {
    TriangleRenderer.renderPropertyImage(mesh,
      renderParameter.pointShader,
      property,
      renderParameter.imageSize.width,
      renderParameter.imageSize.height
    )
  }

}


case class SpecularSphericalHarmonicsLight(coefficients: IndexedSeq[EuclideanVector[_3D]])  {
  require(coefficients.isEmpty || SphericalHarmonics.totalCoefficients(bands) == coefficients.length, "invalid length of coefficients to build SphericalHarmonicsLight")

  def shader(worldMesh: ColorNormalMesh3D, eyePosition: Point[_3D],  specular: SurfacePointProperty[RGBA], specularExp: Double): SphericalHarmonicsSpecularMapsShader = shader(worldMesh, specular, specularExp)

  /** SH shader for a given mesh and this environment map */
  def shader(worldMesh: ColorNormalMesh3D,  specular: SurfacePointProperty[RGBA], specularExp: Double): SphericalHarmonicsSpecularMapsShader = {
    val positionsWorld = SurfacePointProperty[Point[_3D]](worldMesh.shape.triangulation, worldMesh.shape.pointSet.points.toIndexedSeq)

    SphericalHarmonicsSpecularMapsShader(specular,  coefficients, worldMesh.normals, positionsWorld, specularExp )
  }

  /** number of bands */
  def bands: Int = SphericalHarmonics.numberOfBandsForCoefficients(coefficients.size)

}

case class SphericalHarmonicsSpecularMapsShader(specularAlbedo: MeshSurfaceProperty[RGBA],
                                                environmentMap: IndexedSeq[EuclideanVector[_3D]],
                                                normalsWorld: MeshSurfaceProperty[EuclideanVector[_3D]],
                                                positionsWorld: MeshSurfaceProperty[Point[_3D]], specularExp: Double
                                               ) extends PixelShader[RGBA] {
  override def apply(triangleId: TriangleId,
                     worldBCC: BarycentricCoordinates,
                     screenCoordinates: Point[_3D]): RGBA = {
    SphericalHarmonicsSpecularMapsShader.specularPart(positionsWorld(triangleId, worldBCC), normalsWorld(triangleId, worldBCC), specularAlbedo(triangleId, worldBCC), environmentMap, specularExp)
  }
}

object SphericalHarmonicsSpecularMapsShader {
  def specularPart(posWorld: Point[_3D], normalWorld: EuclideanVector[_3D], specularAlbedo: RGBA, environmentMap: IndexedSeq[EuclideanVector[_3D]], specularExp: Double): RGBA = {
    /** BRDF: find outgoing vector for given view vector (reflect) */
    val vecToEye: EuclideanVector3D = (Point3D(0, 0, 0) - posWorld).normalize
    val viewAdjusted = vecToEye * vecToEye.dot(normalWorld)
    val reflected = normalWorld + (normalWorld - viewAdjusted)
    /** Read out value in environment map in out direction for specularity */
    val lightInOutDirection = SphericalHarmonicsLambertShader.shade(specularAlbedo, reflected, environmentMap)
    val specularFactor = math.pow(vecToEye.dot(reflected).toDouble, specularExp)
    specularFactor *: lightInOutDirection
  }
}


