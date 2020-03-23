package faces.apps

import java.io.File

import faces.lib.{AlbedoMoMo, AlbedoMoMoIO}

import scalismo.common.UnstructuredPointsDomain
import scalismo.faces.io.MoMoIO
import scalismo.faces.momo.PancakeDLRGP
import scalismo.geometry.{Point, _3D}
import scalismo.utils.Random

object AlbedoModelBuilder extends App{
  scalismo.initialize()
  val seed = 1024L
  implicit val rnd: Random = Random(seed)

  val bfmAlbedo =   AlbedoMoMoIO.read(new File("albedoModel2020_bfm_albedoPart.h5")).get
  val face12Albedo =  AlbedoMoMoIO.read(new File("albedoModel2020_face12_albedoPart.h5")).get

  val bfmShape =   MoMoIO.read(new File("model2017-1_bfm_nomouth.h5")).get
  val face12Shape =  MoMoIO.read(new File("model2017-1_face12_nomouth.h5")).get

  val bfmCombined = AlbedoMoMo(bfmShape.referenceMesh, bfmShape.neutralModel.shape, bfmAlbedo.neutralModel.diffuseAlbedo, bfmAlbedo.neutralModel.specularAlbedo, bfmShape.expressionModel.get.expression, bfmShape.landmarks)
  val face12Combined = AlbedoMoMo(face12Shape.referenceMesh, face12Shape.neutralModel.shape, face12Albedo.neutralModel.diffuseAlbedo, face12Albedo.neutralModel.specularAlbedo, face12Shape.expressionModel.get.expression, face12Shape.landmarks)


  AlbedoMoMoIO.write(bfmCombined, new File("albedoModel2020_bfm.h5"))
  AlbedoMoMoIO.write(face12Combined, new File("albedoModel2020_face12.h5"))
}
