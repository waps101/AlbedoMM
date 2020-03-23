package faces.apps

import java.io.File

import faces.lib.{AlbedoMoMoIO, AlbedoMoMoRenderer, AlbedoModelFitScript}
import scalismo.utils.Random

object AlbedoModelFit extends App{

  scalismo.initialize()
  val seed = 1024L
  implicit val rnd: Random = Random(seed)


  val model =  AlbedoMoMoIO.read(new File("albedoModel2020_face12.h5")).get
  val renderer = AlbedoMoMoRenderer(model)

  val targetFn = "fitting/Bob_Stoops_0005.png"
  val lmFn = "fitting/Bob_Stoops_0005_face0.tlms"
  val outPutDir = "fitting/results"


  AlbedoModelFitScript.fit(targetFn, lmFn, outPutDir, renderer, true)
}
