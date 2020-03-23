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
 *  This code was adapted from
 * https://github.com/unibas-gravis/scalismo-faces
 * Copyright University of Basel, Graphics and Vision Research Group
 */
package faces.lib

import java.io._

import breeze.linalg.{DenseVector, norm, sum}
import breeze.stats.distributions.Gaussian
import faces.FacesTestSuite
import scalismo.faces.momo.{MoMoCoefficients, PancakeDLRGP}

class AlbedoMoMoTests extends FacesTestSuite {

  def meshDist(mesh1: VertexAlbedoMesh3D, mesh2: VertexAlbedoMesh3D): Double = {
    val shp1: DenseVector[Double] = DenseVector(mesh1.shape.pointSet.points.toIndexedSeq.flatMap(p => IndexedSeq(p.x, p.y, p.z)).toArray)
    val shp2: DenseVector[Double] = DenseVector(mesh2.shape.pointSet.points.toIndexedSeq.flatMap(p => IndexedSeq(p.x, p.y, p.z)).toArray)
    val dif1: DenseVector[Double] = DenseVector(mesh1.diffuseAlbedo.pointData.flatMap(p => IndexedSeq(p.r, p.g, p.b)).toArray)
    val dif2: DenseVector[Double] = DenseVector(mesh2.diffuseAlbedo.pointData.flatMap(p => IndexedSeq(p.r, p.g, p.b)).toArray)
    val spe1: DenseVector[Double] = DenseVector(mesh1.specularAlbedo.pointData.flatMap(p => IndexedSeq(p.r, p.g, p.b)).toArray)
    val spe2: DenseVector[Double] = DenseVector(mesh2.specularAlbedo.pointData.flatMap(p => IndexedSeq(p.r, p.g, p.b)).toArray)
    val shapeDistSq = sum((shp1 - shp2).map(v => v * v))
    val diffuseDistSq = sum((dif1 - dif2).map(v => v * v))
    val specularDistSq = sum((spe1 - spe2).map(v => v * v))
    math.sqrt(shapeDistSq + diffuseDistSq + specularDistSq)
  }

  def coeffsDist(coeffs1: MoMoCoefficients, coeffs2: MoMoCoefficients): Double = {
    require(coeffs1.shape.length == coeffs2.shape.length)
    require(coeffs1.color.length == coeffs2.color.length)
    require(coeffs1.expression.length == coeffs2.expression.length)
    norm(coeffs1.shape - coeffs2.shape) + norm(coeffs1.color - coeffs2.color) + norm(coeffs1.expression - coeffs2.expression)
  }

  describe("A AlbedoMoMo") {

    // Build random PCA model for projection tests and write it to disk
    lazy val randomAlbedoMoMo = randomGridModel(10, 5, 0.1, 5, 5, orthogonalExpressions = false)
    val rf = File.createTempFile("AlbedoMoMo-gravisAlbedoMoMoio-test", ".h5")
    rf.deleteOnExit()
    AlbedoMoMoIO.write(randomAlbedoMoMo, rf)

    // load hdf5 default model
    lazy val fullAlbedoMoMo = AlbedoMoMoIO.read(new File(getClass.getResource("/random-AlbedoMoMo.h5").getPath)).get.expressionModel.get
    lazy val fullShapeCoeffs = for (_ <- 0 until fullAlbedoMoMo.shape.rank) yield Gaussian(0, 1).draw()
    //since coeffs coupled only diffuse
    lazy val fullDiffuseAlbedoCoeffs = for (_ <- 0 until fullAlbedoMoMo.diffuseAlbedo.rank) yield Gaussian(0, 1).draw()
    lazy val fullExpressCoeffs = for (_ <- 0 until fullAlbedoMoMo.expression.rank) yield Gaussian(0, 1).draw()
    lazy val fullAlbedoMoMoCoeffs = MoMoCoefficients(fullShapeCoeffs, fullDiffuseAlbedoCoeffs, fullExpressCoeffs)

    lazy val fullSample = fullAlbedoMoMo.instance(fullAlbedoMoMoCoeffs)

    // PCA model needed for projection tests
    lazy val albedoMoMo = AlbedoMoMoIO.read(rf).get.expressionModel.get

    val AlbedoMoMoPCA = AlbedoMoMo(
      albedoMoMo.referenceMesh,
      PancakeDLRGP(albedoMoMo.shape.gpModel),
      PancakeDLRGP(albedoMoMo.diffuseAlbedo.gpModel),
      PancakeDLRGP(albedoMoMo.specularAlbedo.gpModel),
      PancakeDLRGP(albedoMoMo.expression.gpModel),
      albedoMoMo.landmarks)

    lazy val shapeCoeffs = for (_ <- 0 until albedoMoMo.shape.rank) yield Gaussian(0, 1).draw()
    //since coupled only diffuse
    lazy val diffuseCoeffs = for (_ <- 0 until albedoMoMo.diffuseAlbedo.rank) yield Gaussian(0, 1).draw()
    lazy val expressCoeffs = for (_ <- 0 until albedoMoMo.expression.rank) yield Gaussian(0, 1).draw()
    lazy val AlbedoMoMoCoeffs = MoMoCoefficients(shapeCoeffs, diffuseCoeffs, expressCoeffs)
    lazy val AlbedoMoMoCoeffsNoEx = AlbedoMoMoCoeffs.copy(expression = DenseVector.zeros[Double](0))
    lazy val sample = albedoMoMo.instance(AlbedoMoMoCoeffs)
    lazy val sampleNoEx = albedoMoMo.neutralModel.instance(AlbedoMoMoCoeffsNoEx)

    val distThres = 0.01 * sample.shape.pointSet.numberOfPoints + 0.01 * sample.diffuseAlbedo.triangulation.pointIds.size+ 0.01 * sample.specularAlbedo.triangulation.pointIds.size

    it("can load from disk") {
      albedoMoMo.shape.rank should be > 0
    }

    it("should create shape samples") {
      sample.shape.pointSet.numberOfPoints should be > 0
    }

    it("should create color samples") {
      sample.diffuseAlbedo.triangulation.pointIds.size should be > 0
      sample.specularAlbedo.triangulation.pointIds.size should be > 0
    }

    it("can generate random samples") {
      albedoMoMo.sample().shape.triangulation.pointIds should be (albedoMoMo.referenceMesh.triangulation.pointIds)
    }

    it("can generate random samples (PCA)") {
      AlbedoMoMoPCA.sample().shape.triangulation.pointIds should be (AlbedoMoMoPCA.referenceMesh.triangulation.pointIds)
    }

    it("can create samples with fewer parameters set") {
      val AlbedoMoMoRed = AlbedoMoMoCoeffs.copy(
        shape = DenseVector(AlbedoMoMoCoeffs.shape.toArray.take(1)),
        color = DenseVector(AlbedoMoMoCoeffs.color.toArray.take(1)),
        expression = DenseVector(AlbedoMoMoCoeffs.expression.toArray.take(1))
      )
      val sample = albedoMoMo.instance(AlbedoMoMoRed)
      val AlbedoMoMoRedFull = AlbedoMoMoCoeffs.copy(
        shape = DenseVector(AlbedoMoMoCoeffs.shape.toArray.take(1) ++ Array.fill(albedoMoMo.shape.rank - 1)(0.0)),
        color = DenseVector(AlbedoMoMoCoeffs.color.toArray.take(1) ++ Array.fill(albedoMoMo.shape.rank - 1)(0.0)),
        expression = DenseVector(AlbedoMoMoCoeffs.expression.toArray.take(1) ++ Array.fill(albedoMoMo.expression.rank - 1)(0.0))
      )
      val sampleFull = albedoMoMo.instance(AlbedoMoMoRedFull)
      meshDist(sample, sampleFull) should be < distThres
    }

    it("should creates instances identical to underlying GP model (no noise on instance)") {
      val instPCA = AlbedoMoMoPCA.instance(AlbedoMoMoCoeffs)
      meshDist(instPCA, sample) should be < distThres
    }

    it("should not alter a sample through projection (PCA model only, no expressions)") {
      val projected = AlbedoMoMoPCA.neutralModel.project(sampleNoEx)
      meshDist(projected, sampleNoEx) should be < distThres
    }

    it("should not alter a sample through projection (PCA model only, with expression)") {
      val projected = AlbedoMoMoPCA.project(sample)
      meshDist(projected, sample) should be < distThres
    }

    it("should regularize a sample through projection (closer to mean, no expression)") {
      val projected = albedoMoMo.neutralModel.project(sampleNoEx)
      meshDist(projected, albedoMoMo.neutralModel.mean) should be < meshDist(sampleNoEx, albedoMoMo.neutralModel.mean)
    }

    it("should regularize the coefficients of a sample (closer to mean)") {
      val projCoeffs = albedoMoMo.coefficients(sample)
      val projCoeffPCA = AlbedoMoMoPCA.coefficients(sample)
      norm(projCoeffs.shape) + norm(projCoeffs.expression) should be < (norm(projCoeffPCA.shape) + norm(projCoeffPCA.expression))
    }

    it("should regularize the coefficients of a sample of the neutral model (closer to mean)") {
      val projCoeffs = albedoMoMo.neutralModel.coefficients(sampleNoEx)
      val projCoeffPCA = AlbedoMoMoPCA.neutralModel.coefficients(sampleNoEx)
      norm(projCoeffs.shape) should be < norm(projCoeffPCA.shape)
    }

    it("should yield the same shape coefficients used to draw a sample (PCA model only)") {
      val projCoeffs = AlbedoMoMoPCA.coefficients(sample)
      val pC = albedoMoMo.coefficients(sample)
      norm(projCoeffs.shape - AlbedoMoMoCoeffs.shape) should be < 0.01 * AlbedoMoMoCoeffs.shape.length
      norm(projCoeffs.color - AlbedoMoMoCoeffsNoEx.color) should be < 0.01 * AlbedoMoMoCoeffs.color.length
      norm(projCoeffs.expression - AlbedoMoMoCoeffs.expression) should be < 0.01 * AlbedoMoMoCoeffs.expression.length
    }

    it("should yield proper coefficients for the mean sample") {
      val meanSample = albedoMoMo.mean
      val meanCoeffs = albedoMoMo.coefficients(meanSample)
      norm(meanCoeffs.shape) + norm(meanCoeffs.color) should be < 0.01 * AlbedoMoMoCoeffs.shape.length + 0.01 * AlbedoMoMoCoeffs.color.length
    }

    val f = File.createTempFile("AlbedoMoMo", ".h5")
    f.deleteOnExit()
    AlbedoMoMoIO.write(albedoMoMo, f).get
    val loadedAlbedoMoMo = AlbedoMoMoIO.read(f).get.expressionModel.get

    describe("should save to and load from disk unaltered") {
      it("reference") {
        loadedAlbedoMoMo.referenceMesh should be(albedoMoMo.referenceMesh)
      }

      it("shape") {
        loadedAlbedoMoMo.shape should be(albedoMoMo.shape)
      }

      it("color") {
        loadedAlbedoMoMo.diffuseAlbedo should be(albedoMoMo.diffuseAlbedo)
        loadedAlbedoMoMo.specularAlbedo should be(albedoMoMo.specularAlbedo)
      }

      it("landmarks") {
        loadedAlbedoMoMo.landmarks should be(albedoMoMo.landmarks)
      }

      it("complete AlbedoMoMo") {
        loadedAlbedoMoMo should be(albedoMoMo)
      }
    }

    /*  it("can load model built in c++ from disk") {
        val loadedCPP = AlbedoMoMoIO.read(new File(getClass.getResource("/random-l4.h5").getPath)).get.neutralModel
        loadedCPP.shape.rank shouldBe 19
        loadedCPP.diffuseAlbedo.rank shouldBe 19
        loadedCPP.specularAlbedo.rank shouldBe 19
      }*/

    lazy val reducedAlbedoMoMo = albedoMoMo.truncate(5, 5, 5)
    lazy val reducedShapeCoeffs = for (_ <- 0 until reducedAlbedoMoMo.shape.rank) yield Gaussian(0, 1).draw()
    //since they are coupled only diffuse
    lazy val reducedDiffuseAlbedoCoeffs = for (_ <- 0 until reducedAlbedoMoMo.diffuseAlbedo.rank) yield Gaussian(0, 1).draw()
    lazy val reducedExpressCoeffs = for (_ <- 0 until reducedAlbedoMoMo.expression.rank) yield Gaussian(0, 1).draw()
    lazy val reducedAlbedoMoMoCoeffs = MoMoCoefficients(reducedShapeCoeffs, reducedDiffuseAlbedoCoeffs, reducedExpressCoeffs)
    lazy val reducedAlbedoMoMoCoeffsNoEx = MoMoCoefficients(reducedShapeCoeffs, reducedDiffuseAlbedoCoeffs, IndexedSeq.empty)
    lazy val reducedSample = reducedAlbedoMoMo.instance(reducedAlbedoMoMoCoeffs)
    lazy val reducedSampleNoEx = reducedAlbedoMoMo.neutralModel.instance(reducedAlbedoMoMoCoeffs)


    it("can be reduced to 5 shape components") {
      reducedAlbedoMoMo.shape.rank should be(5)
    }

    it("can be reduced to 5 color components") {
      reducedAlbedoMoMo.diffuseAlbedo.rank should be(5)
      reducedAlbedoMoMo.specularAlbedo.rank should be(5)
    }

    it("can be reduced to 5 expression components") {
      reducedAlbedoMoMo.expression.rank should be(5)
    }

    it("should create shape samples (reduced)") {
      reducedSample.shape.pointSet.numberOfPoints should be > 0
    }

    it("should create color samples (reduced)") {
      reducedSample.diffuseAlbedo.pointData.length should be > 0
      reducedSample.specularAlbedo.pointData.length should be > 0
    }

    it("should not alter a sample through projection (reduced)") {
      val projected = reducedAlbedoMoMo.project(reducedSample)
      meshDist(projected, reducedSample) should be < 0.1 * reducedSample.shape.pointSet.numberOfPoints + 0.01 * reducedSample.diffuseAlbedo.pointData.length + 0.01 * reducedSample.specularAlbedo.pointData.length
    }

    //    it("should yield the same coefficients used to draw a sample (reduced)") {
    //      val projCoeffs = reducedAlbedoMoMo.coefficients(reducedSample)
    //      coeffsDist(projCoeffs, reducedAlbedoMoMoCoeffs) should be < 0.01 * reducedAlbedoMoMoCoeffs.shape.length + 0.01 * reducedAlbedoMoMoCoeffs.color.length + 0.01 * reducedAlbedoMoMoCoeffs.expression.length
    //    }

    it("can be written to disk and be read again (and be equal)") {
      val f = File.createTempFile("reduced-model", ".h5")
      f.deleteOnExit()
      AlbedoMoMoIO.write(albedoMoMo, f).get
      val loadedAlbedoMoMo = AlbedoMoMoIO.read(f).get
      loadedAlbedoMoMo should be(albedoMoMo)
    }
    /*
        it("supports loading using an URI") {
          val modelFile = new File(getClass.getResource("/random-AlbedoMoMo.h5").getPath)
          val AlbedoMoMo = AlbedoMoMoIO.read(modelFile, "").get
          val uri = modelFile.toURI
          val uriAlbedoMoMo = AlbedoMoMoIO.read(uri).get
          uriAlbedoMoMo shouldBe albedoMoMo
        }

        it("supports cached loading using an URI") {
          val uri = new File(getClass.getResource("/random-AlbedoMoMo.h5").getPath).toURI
          val uriAlbedoMoMo1 = AlbedoMoMoIO.read(uri).get
          val uriAlbedoMoMo2 = AlbedoMoMoIO.read(uri).get
          // ensure same model instance: caching serves the identical instance twice
          assert(uriAlbedoMoMo1 eq uriAlbedoMoMo2)
        }
    */
    it("it prevents drawing an instance with an empty parameter vector") {
      val emptyVector = MoMoCoefficients(IndexedSeq.empty, IndexedSeq.empty, IndexedSeq.empty)
      intercept[IllegalArgumentException](albedoMoMo.instance(emptyVector))
    }

    it("can handle an empty expression model") {
      val neutralCoeffs = AlbedoMoMoCoeffs.copy(expression = DenseVector.zeros(0))
      noException should be thrownBy albedoMoMo.neutralModel.instance(neutralCoeffs)
    }
  }
}