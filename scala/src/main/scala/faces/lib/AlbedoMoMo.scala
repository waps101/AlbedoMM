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

/* This is basically the same as the usual MoMo from scalismo-faces.
 * MoMoCoefficients stay the same, but each instance has a diffuse and
 * specular part instead of just a color.
 */



import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Gaussian
import faces.lib
import scalismo.color.{RGB, RGBA}
import scalismo.common._
import scalismo.faces.momo.{MoMoCoefficients, PancakeDLRGP}
import scalismo.geometry._
import scalismo.mesh.{TriangleMesh3D, _}
import scalismo.statisticalmodel._
import scalismo.utils.Random

/** 3D Morphable Model with shape, diffuseAlbedo and expressions */
trait AlbedoMoMo {
  /** reference of Morphable Model, defines registration and triangulation */
  def referenceMesh: TriangleMesh3D

  /** all landmarks defined on this model, points on reference mesh */
  def landmarks: Map[String, Landmark[_3D]] = Map.empty[String, Landmark[_3D]]

  /** returns true if this model has a non-empty expression part */
  def hasExpressions: Boolean

  /** construct the neutral (identity) model without expressions */
  def neutralModel: AlbedoMoMoBasic

  def expressionModel: Option[AlbedoMoMoExpress]

  /**
   * Returns the mean VertexAlbedoMesh3D of the model.
   *
   * @return the mean
   */
  def mean: VertexAlbedoMesh3D

  /**
   * Generate model instance described by coefficients.
   *
   * @param coefficients model coefficients
   * @return model instance as mesh with per vertex diffuseAlbedo
   */
  def instance(coefficients: MoMoCoefficients): VertexAlbedoMesh3D

  /**
   * A fast method to evaluate a model instance at a specific point.
   *
   * @param coefficients model coefficients
   * @param pid          point-id
   * @return shape and diffuseAlbedo values at point-id
   */
  def instanceAtPoint(coefficients: MoMoCoefficients, pid: PointId): (Point[_3D], RGB, RGB)

  /**
   * Calculates the parameters of the model representation of the sample.
   *
   * @param sample the sample (shape and diffuseAlbedo)
   * @return the model coefficients
   */
  def coefficients(sample: VertexAlbedoMesh3D): MoMoCoefficients

  /**
   * Projection operator in model space.
   *
   * @param sample the sample (shape and diffuseAlbedo)
   * @return the model reconstruction of the sample
   */
  def project(sample: VertexAlbedoMesh3D): VertexAlbedoMesh3D = instance(coefficients(sample))

  /**
   * Draw a sample from the model.
   *
   * @return a sample (shape and diffuseAlbedo)
   */
  def sample()(implicit rnd: Random): VertexAlbedoMesh3D

  /**
   * Draw a set of coefficients from the prior of the statistical model.
   *
   * @return a set of coefficients to generate a random sample.
   */
  def sampleCoefficients()(implicit rnd: Random): MoMoCoefficients

  /**
   * Returns the same model but with exchanged or added landmarks.
   *
   * @param landmarksMap Map of named landmarks.
   * @return Same model but with exchanged landmarks.
   */
  def withLandmarks(landmarksMap: Map[String, Landmark[_3D]]): AlbedoMoMo

  /**
   * Test if the model has some landmarks.
   *
   * @return true if the model has at least one landmark, false otherwise
   */
  def hasLandmarks: Boolean = landmarks.nonEmpty

  /**
   * Returns the PointId for a given landmark name.
   *
   * @param id name ot the landmark
   * @return return the PointId of the landmark
   */
  def landmarkPointId(id: String): Option[PointId] = {
    for {
      lm <- landmarks.get(id)
      id <- referenceMesh.pointSet.pointId(lm.point)
    } yield id
  }

  /** get all landmarks expressed thruogh PointIds */
  def landmarksWithPointIds: Map[String, Option[PointId]] = landmarks.map{case (id, lm) => id -> referenceMesh.pointSet.pointId(lm.point) }

  /** get all landmarks expressed thruogh PointIds, use closest point for each landmarks (careful!) */
  def landmarksWithClosestPointIds: Map[String, PointId] = landmarks.map{case (id, lm) => id -> referenceMesh.pointSet.findClosestPoint(lm.point).id}

  /** pad a coefficient vector if it is too short, basis with single vector */
  def padCoefficients(momoCoeff: MoMoCoefficients): MoMoCoefficients

  /** get a coefficients vector describing the mean, has proper dimensions */
  def zeroCoefficients: MoMoCoefficients
}

object AlbedoMoMo {
  /** create a Morphable Model */
  def apply(referenceMesh: TriangleMesh3D,
            shape: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], Point[_3D]],
            diffuseAlbedo: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], RGB],
            specularAlbedo: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], RGB],
            expression: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]],
            landmarks: Map[String, Landmark[_3D]]): AlbedoMoMoExpress = {
    AlbedoMoMoExpress(referenceMesh, shape, diffuseAlbedo, specularAlbedo, expression, landmarks)
  }

  /** create a Morphable Model without expressions */
  def apply(referenceMesh: TriangleMesh3D,
            shape: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], Point[_3D]],
            diffuseAlbedo: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], RGB],
            specularAlbedo: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], RGB],
            landmarks: Map[String, Landmark[_3D]]): AlbedoMoMoBasic = {
    AlbedoMoMoBasic(referenceMesh, shape, diffuseAlbedo, specularAlbedo, landmarks)
  }

  /** create a Morphable Model */
  def apply(referenceMesh: TriangleMesh3D,
            shape: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], Point[_3D]],
            diffuseAlbedo: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], RGB],
            specularAlbedo: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], RGB],
            expression: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]]): AlbedoMoMoExpress = {
    AlbedoMoMoExpress(referenceMesh, shape, diffuseAlbedo,specularAlbedo, expression, Map.empty)
  }

  /** create a Morphable Model without expressions */
  def apply(referenceMesh: TriangleMesh3D,
            shape: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], Point[_3D]],
            diffuseAlbedo: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], RGB],
            specularAlbedo: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], RGB]): AlbedoMoMoBasic = {
    AlbedoMoMoBasic(referenceMesh, shape, diffuseAlbedo, specularAlbedo, Map.empty)
  }

  /**
   * Builds a MoMo from a scalismo StatisticalMeshModel and a diffuse and specular albedo GP.
   *
   * @param shape scalismo.statisticalmodel.StatisticalMeshModel for the shape
   * @param diffuseAlbedo DLRGP model for the diffuse albedo
   * @param specularAlbedo DLRGP model for the specular albedo
   * @return New MoMo with the statistics of the two model.
   */
  def fromStatisticalMeshModel(shape: StatisticalMeshModel,
                               diffuseAlbedo: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], RGB],
                               specularAlbedo: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], RGB],
                               shapeNoiseVariance: Double = 0.0): AlbedoMoMoBasic = {
    val shapeModel = PancakeDLRGP(AlbedoModelHelpers.vectorToPointDLRGP(shape.gp, shape.referenceMesh), shapeNoiseVariance)
    AlbedoMoMoBasic(shape.referenceMesh, shapeModel, diffuseAlbedo, specularAlbedo)
  }

  /**
   * Build 3d Morphable Model from registered samples.
   *
   * @param reference          the reference mesh
   * @param samplesShape       shape samples in dense correspondence with reference
   * @param samplesDiffuseAlbedo       diffuse albedo samples in dense correspondence with reference
   * @param samplesSpecularAlbedo      specular albedo samples in dense correspondence with reference
   * @param shapeNoiseVariance spherical noise term in PPCA
   * @param diffuseAlbedoNoiseVariance spherical noise term in PPCA
   * @param specularAlbedoNoiseVariance spherical noise term in PPCA
   * @return
   */
  def buildFromRegisteredSamples(reference: TriangleMesh3D,
                                 samplesShape: IndexedSeq[VertexColorMesh3D],
                                 samplesDiffuseAlbedo: IndexedSeq[VertexColorMesh3D],
                                 samplesSpecularAlbedo: IndexedSeq[VertexColorMesh3D],
                                 shapeNoiseVariance: Double,
                                 diffuseAlbedoNoiseVariance: Double,
                                 specularAlbedoNoiseVariance: Double): AlbedoMoMoBasic = {

    require(samplesShape.nonEmpty, "MoMo needs shape samples (>0)")
    require(samplesDiffuseAlbedo.nonEmpty, "MoMo needs diffuseAlbedo samples (>0)")
    require(samplesSpecularAlbedo.nonEmpty, "MoMo needs specular albedo samples (>0)")
    require(samplesShape.forall(e => e.shape.pointSet.numberOfPoints == reference.pointSet.numberOfPoints), "MoMo samples must be compatible with reference")
    require(samplesDiffuseAlbedo.forall(e => e.shape.pointSet.numberOfPoints == reference.pointSet.numberOfPoints), "MoMo samples must be compatible with reference")
    require(samplesSpecularAlbedo.forall(e => e.shape.pointSet.numberOfPoints == reference.pointSet.numberOfPoints), "MoMo samples must be compatible with reference")
    val domain = reference.pointSet

    val shapeSamples = samplesShape.map { (sample: VertexColorMesh3D) => DiscreteField[_3D, UnstructuredPointsDomain[_3D], Point[_3D]](domain, sample.shape.pointSet.points.toIndexedSeq) }
    val diffuseAlbedoSamples = samplesDiffuseAlbedo.map { (sample: VertexColorMesh3D) => DiscreteField[_3D, UnstructuredPointsDomain[_3D], RGB](domain, sample.color.pointData.map{_.toRGB}) }
    val specularAlbedoSamples = samplesSpecularAlbedo.map { (sample: VertexColorMesh3D) => DiscreteField[_3D, UnstructuredPointsDomain[_3D], RGB](domain, sample.color.pointData.map{_.toRGB}) }


    val shapeModel = AlbedoModelHelpers.createUsingPPCA[_3D, UnstructuredPointsDomain[_3D], Point[_3D]](domain, shapeSamples, shapeNoiseVariance)
    val diffuseAlbedoModel = AlbedoModelHelpers.createUsingPPCA[_3D, UnstructuredPointsDomain[_3D], RGB](domain, diffuseAlbedoSamples, diffuseAlbedoNoiseVariance)
    val specularAlbedoModel = AlbedoModelHelpers.createUsingPPCA[_3D, UnstructuredPointsDomain[_3D], RGB](domain,specularAlbedoSamples, specularAlbedoNoiseVariance)

    lib.AlbedoMoMo(reference, shapeModel, diffuseAlbedoModel, specularAlbedoModel)
  }

  /**
   * Keep an expression scan together with its corresponding neutral scan
   **/
  case class NeutralWithExpression(neutral: VertexAlbedoMesh3D, expression: VertexAlbedoMesh3D)

  /**
   * Build a 3D Morphable Model with expressions from registered samples.
   *
   * @param reference                  the reference mesh
   * @param samplesShape               shape samples in dense correspondence with reference
   * @param samplesDiffuseAlbedo       diffuse albedo samples in dense correspondence with reference
   * @param samplesSpecularAlbedo      specular albedo samples in dense correspondence with reference
   * @param samplesExpression          expression model samples, consist of a neutral and an expression sample
   * @param shapeNoiseVariance         spherical noise term in PPCA
   * @param diffuseAlbedoNoiseVariance spherical noise term in PPCA
   * @param specularAlbedoNoiseVariance spherical noise term in PPCA
   * @param expressionNoiseVariance    spherical noise term in PPCA
   * @return
   */
  def buildFromRegisteredSamples(reference: TriangleMesh3D,
                                 samplesShape: IndexedSeq[VertexColorMesh3D],
                                 samplesDiffuseAlbedo: IndexedSeq[VertexColorMesh3D],
                                 samplesSpecularAlbedo: IndexedSeq[VertexColorMesh3D],
                                 samplesExpression: IndexedSeq[NeutralWithExpression],
                                 shapeNoiseVariance: Double,
                                 diffuseAlbedoNoiseVariance: Double,
                                 specularAlbedoNoiseVariance: Double,
                                 expressionNoiseVariance: Double)
  : AlbedoMoMoExpress = {

    require(samplesShape.nonEmpty, "MoMo needs shape samples (>0)")
    require(samplesDiffuseAlbedo.nonEmpty, "MoMo needs diffuse albedo samples (>0)")
    require(samplesSpecularAlbedo.nonEmpty, "MoMo needs specular albedo samples (>0)")
    require(samplesExpression.nonEmpty, "MoMo needs expression samples (>0)")
    require(samplesShape.forall(e => e.shape.pointSet.numberOfPoints == reference.pointSet.numberOfPoints), "MoMo samples must be compatible with reference")
    require(samplesDiffuseAlbedo.forall(e => e.shape.pointSet.numberOfPoints == reference.pointSet.numberOfPoints), "MoMo samples must be compatible with reference")
    require(samplesSpecularAlbedo.forall(e => e.shape.pointSet.numberOfPoints == reference.pointSet.numberOfPoints), "MoMo samples must be compatible with reference")
    require(samplesExpression.forall(e => e.neutral.shape.pointSet.numberOfPoints == reference.pointSet.numberOfPoints), "Expression/Neutral samples must be compatible with reference")
    require(samplesExpression.forall(e => e.expression.shape.pointSet.numberOfPoints == reference.pointSet.numberOfPoints), "Expression samples must be compatible with reference")

    val domain = reference.pointSet

    val shapeSamples = samplesShape.map { (sample: VertexColorMesh3D) => DiscreteField[_3D, UnstructuredPointsDomain[_3D], Point[_3D]](domain, sample.shape.pointSet.points.toIndexedSeq) }
    val diffuseAlbedoSamples = samplesDiffuseAlbedo.map { (sample: VertexColorMesh3D) => DiscreteField[_3D, UnstructuredPointsDomain[_3D], RGB](domain, sample.color.pointData.map{_.toRGB}) }
    val specularAlbedoSamples = samplesSpecularAlbedo.map { (sample: VertexColorMesh3D) => DiscreteField[_3D, UnstructuredPointsDomain[_3D], RGB](domain, sample.color.pointData.map{_.toRGB}) }
    val expressionSamples = samplesExpression.map { case NeutralWithExpression(neutral, exp) =>
      val difference = domain.pointIds.map { pointId => exp.shape.pointSet.point(pointId) - neutral.shape.pointSet.point(pointId) }
      DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](domain, difference.toIndexedSeq)
    }

    val shapeModel = AlbedoModelHelpers.createUsingPPCA[_3D, UnstructuredPointsDomain[_3D], Point[_3D]](domain, shapeSamples, shapeNoiseVariance)
    val diffuseAlbedoModel = AlbedoModelHelpers.createUsingPPCA[_3D, UnstructuredPointsDomain[_3D], RGB](domain, diffuseAlbedoSamples, diffuseAlbedoNoiseVariance)
    val specularAlbedoModel = AlbedoModelHelpers.createUsingPPCA[_3D, UnstructuredPointsDomain[_3D], RGB](domain,specularAlbedoSamples, specularAlbedoNoiseVariance)
    val expressionModel = AlbedoModelHelpers.createUsingPPCA[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](domain, expressionSamples, expressionNoiseVariance)



    lib.AlbedoMoMo(
      reference,
      shapeModel,
      diffuseAlbedoModel,
      specularAlbedoModel,
      expressionModel
    )
  }
}

/**
 * 3d Morphable Model implementation, includes facial expression
 *
 * The model consists of a shape and diffuse and specular albedo model together with noise estimates.
 * Both models are defined over points of the reference Mesh. The models are
 * spherical PPCA models.
 *
 * @param referenceMesh      reference of the model
 * @param shape              the shape model
 * @param diffuseAlbedo              the diffuse albedo model
 * @param specularAlbedo              the specular albedo model
 */
case class AlbedoMoMoExpress(override val referenceMesh: TriangleMesh3D,
                             shape: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], Point[_3D]],
                             diffuseAlbedo: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], RGB],
                             specularAlbedo: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], RGB],
                             expression: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]],
                             override val landmarks: Map[String, Landmark[_3D]])
  extends AlbedoMoMo {

  require(shape.domain == diffuseAlbedo.domain, "shape and diffusea albedo model do not have a matching domain")
  require(shape.domain == specularAlbedo.domain, "shape and specular albedo model do not have a matching domain")
  require(referenceMesh.pointSet == shape.domain, "reference domain does not match model domain")
  require(referenceMesh.pointSet == expression.domain, "expression model does not have a matching domain")

  def hasExpressions: Boolean = true

  override lazy val neutralModel: AlbedoMoMoBasic = AlbedoMoMoBasic(referenceMesh, shape, diffuseAlbedo, specularAlbedo, landmarks)

  override def expressionModel: Option[AlbedoMoMoExpress] = Some(this)

  /**
   * Returns the mean VertexAlbedoMesh3D of the model.
   *
   * @return the mean
   */
  def mean: VertexAlbedoMesh3D = {
    lib.VertexAlbedoMesh3D(
      discreteFieldToShape(shape.mean, expression.mean),
      discreteFieldToColor(diffuseAlbedo.mean),
      discreteFieldToColor(specularAlbedo.mean)
    )
  }

  /**
   * Generate model instance described by coefficients.
   *
   * @param coefficients model coefficients
   * @return model instance as mesh with per vertex diffuse and specular albedo
   */
  def instance(coefficients: MoMoCoefficients): VertexAlbedoMesh3D = {
    val MoMoCoefficients(shapeCoefficients, colorCoefficients, expressCoefficients) = padCoefficients(coefficients)
    lib.VertexAlbedoMesh3D(
      discreteFieldToShape(shape.instance(shapeCoefficients), expression.instance(expressCoefficients)),
      discreteFieldToColor(diffuseAlbedo.instance(colorCoefficients)),
      discreteFieldToColor(specularAlbedo.instance(colorCoefficients))
    )
  }

  /**
   * A fast method to evaluate a model instance at a specific point.
   *
   * @param coefficients model coefficients
   * @param pid          point-id
   * @return shape and diffuse and specular albedo values at point-id
   */
  def instanceAtPoint(coefficients: MoMoCoefficients, pid: PointId): (Point[_3D], RGB, RGB) = {
    val MoMoCoefficients(shapeCoefficients, colorCoefficients, expressCoefficients) = padCoefficients(coefficients)
    val point = shape.instanceAtPoint(shapeCoefficients, pid) + expression.instanceAtPoint(expressCoefficients, pid)
    val diffuseAlbedoAtPoint = diffuseAlbedo.instanceAtPoint(colorCoefficients, pid)
    val specularAlbedoAtPoint = specularAlbedo.instanceAtPoint(colorCoefficients, pid)
    (point, diffuseAlbedoAtPoint, specularAlbedoAtPoint)
  }

  /**
   * Calculates the parameters of the model representation of the sample.
   *
   * @param sample the sample (shape and diffuseAlbedo)
   * @return the model coefficients
   */
  override def coefficients(sample: VertexAlbedoMesh3D): MoMoCoefficients = {

    //TODO: This only projects the diffuse albedo, since the parameters are coupled it should not make a difference, but it would be nice to check
    require(sample.shape.pointSet.numberOfPoints == referenceMesh.pointSet.numberOfPoints, "mesh to project does not have the same amount of points as model")

    // diffuseAlbedo
    val diffuseAlbedoCoeffs = diffuseAlbedo.coefficients(colorToDiscreteField(sample.diffuseAlbedo))

    // composite shape with precalculation of matrices
    // u = f(alpha_N, alpha_E) = mu_N + mu_E + M_N * alpha_N + M_E * alpha_E + epsilon_N + epsilon_N
    // mu_alpha|u = ( (sigma_N^2 + sigma_E^2)*I + W_tilde )^-1 * [W_N | W_E]^T * (u - mu_N - mu_E)
    // W_tilde = [ W_N^T * W_N    W_N^T * W_E ; W_E^T * W_N    W_E^T * W_E ]
    // sigma^2 = [ sigma_N^2 sigma_E^2 ]
    // sigma_N^2: number of elements corresponding to number of columns of M_N
    // sigma_E^2: number of elements corresponding to number of columns of M_E

    val pointIds = sample.shape.pointSet.pointIds.toIndexedSeq
    val vertexDim = shape.vectorizer.dim
    val shapeVectorLength = vertexDim * referenceMesh.pointSet.numberOfPoints

    val shapeMu = DenseVector(shape.mean.data.toArray)
    val expressMu = DenseVector(expression.mean.data.toArray)

    val sampleVec = DenseVector.zeros[Double](shapeVectorLength)
    val shapeMuVec = DenseVector.zeros[Double](shapeVectorLength)
    val expressMuVec = DenseVector.zeros[Double](shapeVectorLength)

    //generates vectors from data (calls vectorizer for every point on reference.)
    pointIds.map { pointId =>
      val value = sample.shape.pointSet.point(pointId)
      val range = pointId.id * vertexDim until (pointId.id + 1) * vertexDim
      val vV: DenseVector[Double] = shape.vectorizer.vectorize(value)
      sampleVec(range) := vV
      val sM: DenseVector[Double] = shape.vectorizer.vectorize(shapeMu(pointId.id))
      shapeMuVec(range) := sM
      val sE: DenseVector[Double] = expression.vectorizer.vectorize(expressMu(pointId.id))
      expressMuVec(range) := sE
    }

    val n_S = shape.rank
    val n_E = expression.rank

    val coeffs_SE = coefficientsMatrices.wTildeNoiseInv * coefficientsMatrices.wT * (sampleVec - shapeMuVec - expressMuVec )

    val shapeCoeffs = coeffs_SE(0 until n_S)
    val expressCoeffs = coeffs_SE(n_S until n_E + n_S)

    MoMoCoefficients(shapeCoeffs, diffuseAlbedoCoeffs, expressCoeffs)
  }

  private case class CoeffsMatrices(wTildeNoiseInv: DenseMatrix[Double], wT: DenseMatrix[Double])

  private lazy val coefficientsMatrices: CoeffsMatrices = {
    // composite shape
    // u = f(alpha_N, alpha_E) = mu_N + mu_E + M_N * alpha_N + M_E * alpha_E + epsilon_N + epsilon_N
    // mu_alpha|u = ( (sigma_N^2 + sigma_E^2)*I + W_tilde )^-1 * [W_N | W_E]^T * (u - mu_N - mu_E)
    // W_tilde = [ W_N^T * W_N    W_N^T * W_E ; W_E^T * W_N    W_E^T * W_E ]
    // sigma^2 = [ sigma_N^2 sigma_E^2 ]
    // sigma_N^2: number of elements corresponding to number of columns of M_N
    // sigma_E^2: number of elements corresponding to number of columns of M_E

    val pointIds = referenceMesh.pointSet.pointIds.toIndexedSeq

    val vertexDim = shape.vectorizer.dim

    val shapeVectorLength = vertexDim * referenceMesh.pointSet.numberOfPoints

    val shapeBasis = shape.basisMatrixScaled

    val expressBasis = expression.basisMatrixScaled

    val shapeMu = DenseVector(shape.mean.data.toArray)
    val expressMu = DenseVector(expression.mean.data.toArray)

    val shapeMuVec = DenseVector.zeros[Double](shapeVectorLength)
    val expressMuVec = DenseVector.zeros[Double](shapeVectorLength)

    //generates vectors from data (calls vectorizer for every point on reference.)
    pointIds.map { pointId =>
      val range = pointId.id * vertexDim until (pointId.id + 1) * vertexDim
      val sM: DenseVector[Double] = shape.vectorizer.vectorize(shapeMu(pointId.id))
      shapeMuVec(range) := sM
      val sE: DenseVector[Double] = expression.vectorizer.vectorize(expressMu(pointId.id))
      expressMuVec(range) := sE
    }

    // W_tilde = [W_tilde1 W_tilde2]
    //           [W_tilde3 W_tilde4]
    // sizes: W_tilde1: n_s x n_s
    //        W_tilde2: n_s x n_e
    //        W_tilde3: n_e x n_s
    //        W_tilde4: n_e x n_e
    //        n_s:      PCABasisPart_s.cols()
    //        n_e:      PCABasisPart_e.cols()

    val n_S = shapeBasis.cols
    val n_E = expressBasis.cols
    val m_S = shapeBasis.rows

    val w_tilde = DenseMatrix.zeros[Double](n_S + n_E, n_S + n_E)

    // W_tilde1
    w_tilde(0 until n_S, 0 until n_S) := shapeBasis.t * shapeBasis
    // W_tilde2
    w_tilde(0 until n_S, n_S until n_S + n_E) := shapeBasis.t * expressBasis
    // W_tilde3
    w_tilde(n_S until n_E + n_S, 0 until n_S) := expressBasis.t * shapeBasis
    // W_tilde4
    w_tilde(n_S until n_E + n_S, n_S until n_E + n_S) := expressBasis.t * expressBasis

    // add sigma^2*I
    val w_tilde_noise = w_tilde + DenseMatrix.eye[Double](n_S + n_E) * (shape.noiseVariance + expression.noiseVariance)

    // WT = [ W_n^T ]
    //      [ W_e^T ]
    val wt = DenseMatrix.zeros[Double](n_S + n_E, m_S)
    wt(0 until n_S, 0 until m_S) := shapeBasis.t
    wt(n_S until n_S + n_E, 0 until m_S) := expressBasis.t

    CoeffsMatrices(breeze.linalg.inv(w_tilde_noise), wt)
  }


  /**
   * Draw a sample from the model.
   *
   * @return a sample (shape and diffuse and specular albedo)
   */
  override def sample()(implicit rnd: Random): VertexAlbedoMesh3D = {

    // draw coefficients for albedo since specular and diffuse are coupled
    val standardNormal = Gaussian(0, 1)(rnd.breezeRandBasis)
    val rank = diffuseAlbedo.rank
    val coeffs = standardNormal.sample(rank)

    lib.VertexAlbedoMesh3D(
      discreteFieldToShape(shape.gpModel.sample(), expression.gpModel.sample()),
      discreteFieldToColor(diffuseAlbedo.gpModel.instance(DenseVector(coeffs.toArray))),
      discreteFieldToColor(specularAlbedo.gpModel.instance(DenseVector(coeffs.toArray)))
    )
  }


  /**
   * Draw a set of coefficients from the prior of the statistical model.
   *
   * @return a set of coefficients to generate a random sample.
   */
  override def sampleCoefficients()(implicit rnd: Random): MoMoCoefficients = {
    MoMoCoefficients(
      shape.coefficientsDistribution.sample(),
      diffuseAlbedo.coefficientsDistribution.sample(),
      expression.coefficientsDistribution.sample())
  }

  /**
   * Returns the same model but with exchanged or added landmarks.
   *
   * @param landmarksMap Map of named landmarks.
   * @return Same model but with exchanged landmarks.
   */
  override def withLandmarks(landmarksMap: Map[String, Landmark[_3D]]): AlbedoMoMo = AlbedoMoMoExpress(referenceMesh, shape, diffuseAlbedo, specularAlbedo, expression, landmarksMap)

  /** pad a coefficient vector if it is too short, basis with single vector */
  override def padCoefficients(momoCoeff: MoMoCoefficients): MoMoCoefficients = {
    def pad(coefficients: DenseVector[Double], rank: Int): DenseVector[Double] = {
      require(coefficients.length <= rank, "too many coefficients for model")
      require(rank == 0 || coefficients.length > 0, "coefficient vector cannot be empty")
      if (coefficients.length == rank)
        coefficients
      else
        DenseVector(coefficients.toArray ++ Array.fill(rank - coefficients.length)(0.0))
    }

    momoCoeff.copy(
      shape = pad(momoCoeff.shape, shape.rank),
      color = pad(momoCoeff.color, diffuseAlbedo.rank),
      expression = pad(momoCoeff.expression, expression.rank)
    )
  }

  override def zeroCoefficients: MoMoCoefficients = MoMoCoefficients(
    DenseVector.zeros[Double](shape.rank),
    DenseVector.zeros[Double](diffuseAlbedo.rank),
    DenseVector.zeros[Double](expression.rank)
  )

  /**
   * Reduces the rank of the model. Drops components only (pure truncation, no noise recalculation)
   *
   * @param shapeComps   Number of shape components to keep.
   * @param colorComps   Number of diffuse and specular albedo components to keep.
   * @param expressComps Number of expression components to keep.
   * @return Reduced model.
   */
  def truncate(shapeComps: Int, colorComps: Int, expressComps: Int): AlbedoMoMoExpress = {
    require(shapeComps >= 0 && shapeComps <= shape.rank, "illegal number of reduced shape components")
    require(colorComps >= 0 && colorComps <= diffuseAlbedo.rank, "illegal number of reduced diffuse albedo components")
    require(colorComps >= 0 && colorComps <= specularAlbedo.rank, "illegal number of reduced specular lbedo components")

    require(expressComps >= 0 && expressComps <= expression.rank, "illegal number of reduced expression components")

    AlbedoMoMoExpress(
      referenceMesh,
      shape.truncate(shapeComps),
      diffuseAlbedo.truncate(colorComps),
      specularAlbedo.truncate(colorComps),
      expression.truncate(expressComps),
      landmarks)
  }

  // converters to deal with discrete fields
  private def discreteFieldToShape(shapeField: DiscreteField[_3D, UnstructuredPointsDomain[_3D], Point[_3D]], expressionField: DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]]): TriangleMesh3D = {
    val points = shapeField.data.zip(expressionField.data).map{case(s, e) => s + e}
    TriangleMesh3D(points, referenceMesh.triangulation)
  }

  private def discreteFieldToColor(colorField: DiscreteField[_3D, UnstructuredPointsDomain[_3D], RGB]): SurfacePointProperty[RGBA] = SurfacePointProperty(referenceMesh.triangulation, colorField.data.map(_.toRGBA))

  private def shapeToDiscreteField(shape: TriangleMesh[_3D]): DiscreteField[_3D, UnstructuredPointsDomain[_3D], Point[_3D]] = DiscreteField(referenceMesh.pointSet, shape.pointSet.points.toIndexedSeq)

  private def colorToDiscreteField(color: SurfacePointProperty[RGBA]): DiscreteField[_3D, UnstructuredPointsDomain[_3D], RGB] = DiscreteField(referenceMesh.pointSet, color.pointData.map(_.toRGB))
}

case class AlbedoMoMoBasic(override val referenceMesh: TriangleMesh3D,
                           shape: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], Point[_3D]],
                           diffuseAlbedo: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], RGB],
                           specularAlbedo: PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], RGB],
                           override val landmarks: Map[String, Landmark[_3D]] = Map.empty[String, Landmark[_3D]])
  extends AlbedoMoMo {

  override def hasExpressions: Boolean = false

  override def neutralModel: AlbedoMoMoBasic = this

  override def expressionModel: Option[AlbedoMoMoExpress] = None

  /**
   * Returns the mean VertexAlbedoMesh3D of the model.
   *
   * @return the mean
   */
  override def mean: VertexAlbedoMesh3D = {
    val shapeInstance = discreteFieldToShape(shape.mean)
    val diffuseAlbedoInstance = discreteFieldToColor(diffuseAlbedo.mean)
    val specularAlbedoInstance = discreteFieldToColor(specularAlbedo.mean)

    lib.VertexAlbedoMesh3D(shapeInstance, diffuseAlbedoInstance, specularAlbedoInstance)
  }

  /**
   * Generate model instance described by coefficients.
   *
   * @param coefficients model coefficients
   * @return model instance as mesh with per vertex diffuse and specular albedo
   */
  override def instance(coefficients: MoMoCoefficients): VertexAlbedoMesh3D = {
    val MoMoCoefficients(shapeCoefficients, colorCoefficients, _) = padCoefficients(coefficients)
    val shapeInstance = discreteFieldToShape(shape.instance(shapeCoefficients))
    val diffuseAlbedoInstance = discreteFieldToColor(diffuseAlbedo.instance(colorCoefficients))
    val specularAlbedoInstance = discreteFieldToColor(specularAlbedo.instance(colorCoefficients))
    lib.VertexAlbedoMesh3D(shapeInstance, diffuseAlbedoInstance, specularAlbedoInstance)
  }

  /**
   * A fast method to evaluate a model instance at a specific point.
   *
   * @param coefficients model coefficients
   * @param pid          point-id
   * @return shape and diffuse and specular albedo values at point-id
   */
  override def instanceAtPoint(coefficients: MoMoCoefficients, pid: PointId): (Point[_3D], RGB, RGB) = {
    val MoMoCoefficients(shapeCoefficients, colorCoefficients, _) = padCoefficients(coefficients)
    val point = shape.instanceAtPoint(shapeCoefficients, pid)
    val diffuseAlbedoAtPoint = diffuseAlbedo.instanceAtPoint(colorCoefficients, pid)
    val specularAlbedoAtPoint = specularAlbedo.instanceAtPoint(colorCoefficients, pid)
    (point, diffuseAlbedoAtPoint, specularAlbedoAtPoint)
  }

  /**
   * Calculates the parameters of the model representation of the sample.
   *
   * @param sample the sample (shape and diffuse and specular albedo)
   * @return the model coefficients
   */
  override def coefficients(sample: VertexAlbedoMesh3D): MoMoCoefficients = {
    //TODO: This only projects the diffuse albedo, since the parameters are coupled it should not make a difference, but it would be nice to check

    val shapeCoeffs = shape.coefficients(shapeToDiscreteField(sample.shape))
    val diffuseAlbedoCoeffs = diffuseAlbedo.coefficients(colorToDiscreteField(sample.diffuseAlbedo))
    val expressCoeffs = DenseVector.zeros[Double](0)
    MoMoCoefficients(shapeCoeffs, diffuseAlbedoCoeffs, expressCoeffs)
  }

  /**
   * Draw a sample from the model.
   *
   * @return a sample (shape and diffuse and specular )
   */
  override def sample()(implicit rnd: Random): VertexAlbedoMesh3D = {

    // draw coefficients for albedo since specular and diffuse are coupled
    val standardNormal = Gaussian(0, 1)(rnd.breezeRandBasis)
    val rank = diffuseAlbedo.rank
    val coeffs = standardNormal.sample(rank)

    lib.VertexAlbedoMesh3D(
      discreteFieldToShape(shape.gpModel.sample()),
      discreteFieldToColor(diffuseAlbedo.gpModel.instance(DenseVector(coeffs.toArray))),
      discreteFieldToColor(specularAlbedo.gpModel.instance(DenseVector(coeffs.toArray)))
    )
  }


  /**
   * Draw a set of coefficients from the prior of the statistical model.
   *
   * @return a set of coefficients to generate a random sample.
   */
  override def sampleCoefficients()(implicit rnd: Random): MoMoCoefficients = {
    MoMoCoefficients(
      shape.coefficientsDistribution.sample(),
      diffuseAlbedo.coefficientsDistribution.sample()
    )
  }

  /**
   * Returns the same model but with exchanged or added landmarks.
   *
   * @param landmarksMap Map of named landmarks.
   * @return Same model but with exchanged landmarks.
   */
  override def withLandmarks(landmarksMap: Map[String, Landmark[_3D]]): AlbedoMoMo = AlbedoMoMoBasic(referenceMesh, shape, diffuseAlbedo, specularAlbedo, landmarksMap)

  /** pad a coefficient vector if it is too short, basis with single vector */
  override def padCoefficients(momoCoeff: MoMoCoefficients): MoMoCoefficients = {
    def pad(coefficients: DenseVector[Double], rank: Int): DenseVector[Double] = {
      require(coefficients.length <= rank, "too many coefficients for model")
      require(rank == 0 || coefficients.length > 0, "coefficient vector cannot be empty")
      if (coefficients.length == rank)
        coefficients
      else
        DenseVector(coefficients.toArray ++ Array.fill(rank - coefficients.length)(0.0))
    }

    momoCoeff.copy(
      shape = pad(momoCoeff.shape, shape.rank),
      color = pad(momoCoeff.color, diffuseAlbedo.rank),
      expression = DenseVector.zeros(0)
    )
  }

  override def zeroCoefficients: MoMoCoefficients = MoMoCoefficients(
    DenseVector.zeros[Double](shape.rank),
    DenseVector.zeros[Double](diffuseAlbedo.rank),
    DenseVector.zeros[Double](0)
  )

  /**
   * Reduces the rank of the model. Drops components only (pure truncation)
   *
   * @param shapeComps   Number of shape components to keep.
   * @param colorComps   Number of diffuse and specular albedo components to keep.
   * @return Reduced model.
   */
  def truncate(shapeComps: Int, colorComps: Int): AlbedoMoMoBasic = {
    require(shapeComps >= 0 && shapeComps <= shape.rank, "illegal number of reduced shape components")
    require(colorComps >= 0 && colorComps <= diffuseAlbedo.rank, "illegal number of reduced diffuseAlbedo components")
    require(colorComps >= 0 && colorComps <= specularAlbedo.rank, "illegal number of reduced specularAlbedo components")

    // @todo allow reduction with increasing noise to capture removed components
    AlbedoMoMoBasic(
      referenceMesh,
      shape.truncate(shapeComps),
      diffuseAlbedo.truncate(colorComps),
      specularAlbedo.truncate(colorComps),
      landmarks)
  }

  // converters to deal with discrete fields
  private def discreteFieldToShape(shapeField: DiscreteField[_3D, UnstructuredPointsDomain[_3D], Point[_3D]]): TriangleMesh3D = TriangleMesh3D(shapeField.data, referenceMesh.triangulation)

  private def discreteFieldToColor(colorField: DiscreteField[_3D, UnstructuredPointsDomain[_3D], RGB]): SurfacePointProperty[RGBA] = SurfacePointProperty(referenceMesh.triangulation, colorField.data.map(_.toRGBA))

  private def shapeToDiscreteField(shape: TriangleMesh[_3D]): DiscreteField[_3D, UnstructuredPointsDomain[_3D], Point[_3D]] = DiscreteField(referenceMesh.pointSet, shape.pointSet.points.toIndexedSeq)

  private def colorToDiscreteField(color: SurfacePointProperty[RGBA]): DiscreteField[_3D, UnstructuredPointsDomain[_3D], RGB] = DiscreteField(referenceMesh.pointSet, color.pointData.map(_.toRGB))

}