package notPublic

import java.io.File

import breeze.linalg.{DenseMatrix, DenseVector, qr}
import faces.lib.{AlbedoMoMo, AlbedoMoMoIO}
import scalismo.common.UnstructuredPointsDomain
import scalismo.faces.io.MoMoIO
import scalismo.faces.momo.PancakeDLRGP
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.mesh.TriangleMesh3D
import scalismo.statisticalmodel.AlbedoModelHelpers
import scalismo.utils.Random

object CombineModel extends App{
  scalismo.initialize()
  val seed = 1024L
  implicit val rnd: Random = Random(seed)

  val bfmSpecular = "/media/data/work/albedomodel/pipeline-data/data/modelbuilding/model/yamm2020_bfm_nomouth_Specular.h5"
  val bfmDiffuse = "/media/data/work/albedomodel/pipeline-data/data/modelbuilding/model/yamm2020_bfm_nomouth.h5"

  val bfmS = MoMoIO.read(new File(bfmSpecular)).get
  val bfmD = MoMoIO.read(new File(bfmDiffuse)).get

  val face12Specular = "/media/data/work/albedomodel/pipeline-data/data/modelbuilding/model/yamm2020_face12_nomouth_Specular.h5"
  val face12Diffuse = "/media/data/work/albedomodel/pipeline-data/data/modelbuilding/model/yamm2020_face12_nomouth.h5"

  val face12S = MoMoIO.read(new File(face12Specular)).get
  val face12D = MoMoIO.read(new File(face12Diffuse)).get


  def randomGridShape(reference: TriangleMesh3D, rank: Int =1, sdev: Double = 0.1, noise: Double = 0.05, cols: Int = 5, rows: Int = 5, orthogonalExpressions: Boolean = false)(implicit rnd: Random): PancakeDLRGP[_3D, UnstructuredPointsDomain[_3D], Point[_3D]] = {


    val compN = 3 * cols * rows
    val fullShapeComponents = qr(DenseMatrix.fill[Double](compN, 2 * rank)(rnd.scalaRandom.nextGaussian)).q

    val shapeN = compN
    val shapeMean = DenseVector.fill[Double](shapeN)(rnd.scalaRandom.nextGaussian)
    val shapePCABases = fullShapeComponents(::, 0 until rank)
    val shapeVariance = DenseVector.fill[Double](rank)(rnd.scalaRandom.nextDouble * sdev)
    assert(shapeMean.length == shapePCABases.rows, "rows is not correct")
    assert(shapePCABases.cols == rank, "model is of incorrect rank")
    assert(shapeVariance.length == shapePCABases.cols, "wrong number of variances")
    PancakeDLRGP( AlbedoModelHelpers.buildFrom[_3D, UnstructuredPointsDomain[_3D], Point[_3D]](reference.pointSet, shapeMean, shapeVariance, shapePCABases))
  }

  val bfmA = AlbedoMoMo(bfmD.referenceMesh, randomGridShape(bfmD.referenceMesh), bfmD.neutralModel.color, bfmS.neutralModel.color,  bfmD.landmarks)

  val face12A = AlbedoMoMo(face12D.referenceMesh, randomGridShape(face12D.referenceMesh), face12D.neutralModel.color, face12S.neutralModel.color, face12D.landmarks)

  AlbedoMoMoIO.write(bfmA, new File("/media/data/work/albedomodel/releaseCode/albedoMorphableModel/data/albedoModel2020_bfm_albedoPart.h5"))
  AlbedoMoMoIO.write(face12A, new File("/media/data/work/albedomodel/releaseCode/albedoMorphableModel/data/albedoModel2020_face12_albedoPart.h5"))
}
