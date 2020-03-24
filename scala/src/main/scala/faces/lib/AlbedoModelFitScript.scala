package faces.lib

/*
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This code was adapted from
 * https://github.com/unibas-gravis/basel-face-model-viewer
 * Copyright University of Basel, Graphics and Vision Research Group
 */


import java.io.File

import breeze.numerics.pow
import faces.lib.ParameterProposals.implicits._
import scalismo.color.{RGB, RGBA}
import scalismo.faces.image.PixelImage
import scalismo.faces.io.{PixelImageIO, RenderParameterIO, TLMSLandmarksIO}
import scalismo.faces.parameters.RenderParameter
import scalismo.faces.sampling.face.evaluators.PixelEvaluators._
import scalismo.faces.sampling.face.evaluators.PointEvaluators.IsotropicGaussianPointEvaluator
import scalismo.faces.sampling.face.evaluators.PriorEvaluators.{GaussianShapePrior, GaussianTexturePrior}
import scalismo.faces.sampling.face.evaluators._
import scalismo.faces.sampling.face.loggers._
import scalismo.faces.sampling.face.proposals.ImageCenteredProposal.implicits._
import scalismo.faces.sampling.face.proposals.SphericalHarmonicsLightProposals._
import scalismo.faces.sampling.face.proposals._
import scalismo.faces.sampling.face.{ParametricLandmarksRenderer, ParametricModel}
import scalismo.geometry.{EuclideanVector, EuclideanVector3D, _2D}
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.loggers.BestSampleLogger
import scalismo.sampling.loggers.ChainStateLogger.implicits._
import scalismo.sampling.loggers.ChainStateLoggerContainer.implicits._
import scalismo.sampling.proposals.MixtureProposal.implicits._
import scalismo.sampling.proposals.{MetropolisFilterProposal, MixtureProposal}
import scalismo.sampling.{ProposalGenerator, TransitionProbability}
import scalismo.utils.Random

/* This Fitscript with its evaluators and the proposal distribution follows closely the proposed setting of:

Markov Chain Monte Carlo for Automated Face Image Analysis
Sandro Schonborn, Bernhard Egger, Andreas Morel-Forster and Thomas Vetter
International Journal of Computer Vision 123(2), 160-183 , June 2017
DOI: http://dx.doi.org/10.1007/s11263-016-0967-5

To understand the concepts behind the fitscript and the underlying methods there is a tutorial on:
http://gravis.dmi.unibas.ch/pmm/

 */

object AlbedoModelFitScript {

  /* Collection of all pose related proposals */
  def defaultPoseProposal(lmRenderer: ParametricLandmarksRenderer)(implicit rnd: Random):
  ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {
    import MixtureProposal.implicits._

    val yawProposalC = GaussianRotationProposal(EuclideanVector3D.unitY, 0.75f)
    val yawProposalI = GaussianRotationProposal(EuclideanVector3D.unitY, 0.10f)
    val yawProposalF = GaussianRotationProposal(EuclideanVector3D.unitY, 0.01f)
    val rotationYaw = MixtureProposal(0.1 *: yawProposalC + 0.4 *: yawProposalI + 0.5 *: yawProposalF)

    val pitchProposalC = GaussianRotationProposal(EuclideanVector3D.unitX, 0.75f)
    val pitchProposalI = GaussianRotationProposal(EuclideanVector3D.unitX, 0.10f)
    val pitchProposalF = GaussianRotationProposal(EuclideanVector3D.unitX, 0.01f)
    val rotationPitch = MixtureProposal(0.1 *: pitchProposalC + 0.4 *: pitchProposalI + 0.5 *: pitchProposalF)

    val rollProposalC = GaussianRotationProposal(EuclideanVector3D.unitZ, 0.75f)
    val rollProposalI = GaussianRotationProposal(EuclideanVector3D.unitZ, 0.10f)
    val rollProposalF = GaussianRotationProposal(EuclideanVector3D.unitZ, 0.01f)
    val rotationRoll = MixtureProposal(0.1 *: rollProposalC + 0.4 *: rollProposalI + 0.5 *: rollProposalF)

    val rotationProposal = MixtureProposal(0.5 *: rotationYaw + 0.3 *: rotationPitch + 0.2 *: rotationRoll).toParameterProposal

    val translationC = GaussianTranslationProposal(EuclideanVector(300f, 300f)).toParameterProposal
    val translationF = GaussianTranslationProposal(EuclideanVector(50f, 50f)).toParameterProposal
    val translationHF = GaussianTranslationProposal(EuclideanVector(10f, 10f)).toParameterProposal
    val translationProposal = MixtureProposal(0.2 *: translationC + 0.2 *: translationF + 0.6 *: translationHF)

    val distanceProposalC = GaussianDistanceProposal(500f, compensateScaling = true).toParameterProposal
    val distanceProposalF = GaussianDistanceProposal(50f, compensateScaling = true).toParameterProposal
    val distanceProposalHF = GaussianDistanceProposal(5f, compensateScaling = true).toParameterProposal
    val distanceProposal = MixtureProposal(0.2 *: distanceProposalC + 0.6 *: distanceProposalF + 0.2 *: distanceProposalHF)

    val scalingProposalC = GaussianScalingProposal(0.15f).toParameterProposal
    val scalingProposalF = GaussianScalingProposal(0.05f).toParameterProposal
    val scalingProposalHF = GaussianScalingProposal(0.01f).toParameterProposal
    val scalingProposal = MixtureProposal(0.2 *: scalingProposalC + 0.6 *: scalingProposalF + 0.2 *: scalingProposalHF)

    val poseMovingNoTransProposal = MixtureProposal(rotationProposal + distanceProposal + scalingProposal)
    val centerREyeProposal = poseMovingNoTransProposal.centeredAt("right.eye.corner_outer", lmRenderer).get
    val centerLEyeProposal = poseMovingNoTransProposal.centeredAt("left.eye.corner_outer", lmRenderer).get
    val centerRLipsProposal = poseMovingNoTransProposal.centeredAt("right.lips.corner", lmRenderer).get
    val centerLLipsProposal = poseMovingNoTransProposal.centeredAt("left.lips.corner", lmRenderer).get

    MixtureProposal(centerREyeProposal + centerLEyeProposal + centerRLipsProposal + centerLLipsProposal + 0.2 *: translationProposal)
  }


  /* Collection of all illumination related proposals */
  def defaultIlluminationProposalNoOptimizer()(implicit rnd: Random):
  ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {


    val lightSHPert = SHLightPerturbationProposal(0.001f, fixIntensity = true)
    val lightSHIntensity = SHLightIntensityProposal(0.1f)
    val lightSHBandMixter = SHLightBandEnergyMixer(0.1f)
    val lightSHSpatial = SHLightSpatialPerturbation(0.05f)
    val lightSHColor = SHLightColorProposal(0.01f)

    MixtureProposal(lightSHSpatial + lightSHBandMixter + lightSHIntensity + lightSHPert + lightSHColor).toParameterProposal
  }

  /* Collection of all illumination related proposals */
  def defaultIlluminationProposalNoOptimizerExponentHack(modelRenderer: ParametricModel, target: PixelImage[RGBA])(implicit rnd: Random):
  ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {

    val lightSHPert = SHLightPerturbationProposal(0.001f, fixIntensity = true)
    val lightSHIntensity = SHLightIntensityProposal(0.1f)
    val lightSHBandMixter = SHLightBandEnergyMixer(0.1f)
    val lightSHSpatial = SHLightSpatialPerturbation(0.05f)
    val lightSHColor = SHLightColorProposal(0.01f)

    MixtureProposal(MixtureProposal(lightSHSpatial + lightSHBandMixter + lightSHIntensity + lightSHPert + lightSHColor).toParameterProposal)
  }


  /* Collection of all statistical model (shape, texture) related proposals */
  def neutralMorphableModelProposal(implicit rnd: Random):
  ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {

    val shapeC = GaussianMoMoShapeProposal(0.2f)
    val shapeF = GaussianMoMoShapeProposal(0.1f)
    val shapeHF = GaussianMoMoShapeProposal(0.025f)
    val shapeScaleProposal = GaussianMoMoShapeCaricatureProposal(0.2f)
    val shapeProposal = MixtureProposal(0.1f *: shapeC + 0.5f *: shapeF + 0.2f *: shapeHF + 0.2f *: shapeScaleProposal).toParameterProposal

    val textureC = GaussianMoMoColorProposal(0.2f)
    val textureF = GaussianMoMoColorProposal(0.1f)
    val textureHF = GaussianMoMoColorProposal(0.025f)
    val textureScale = GaussianMoMoColorCaricatureProposal(0.2f)
    val textureProposal = MixtureProposal(0.1f *: textureC + 0.5f *: textureF + 0.2 *: textureHF + 0.2f *: textureScale).toParameterProposal

    MixtureProposal(shapeProposal + textureProposal)
  }

  /* Collection of all statistical model (shape, texture, expression) related proposals */
  def defaultMorphableModelProposal(implicit rnd: Random):
  ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {


    val expressionC = GaussianMoMoExpressionProposal(0.2f)
    val expressionF = GaussianMoMoExpressionProposal(0.1f)
    val expressionHF = GaussianMoMoExpressionProposal(0.025f)
    val expressionScaleProposal = GaussianMoMoExpressionCaricatureProposal(0.2f)
    val expressionProposal = MixtureProposal(0.1f *: expressionC + 0.5f *: expressionF + 0.2f *: expressionHF + 0.2f *: expressionScaleProposal).toParameterProposal


    MixtureProposal(neutralMorphableModelProposal + expressionProposal)
  }

  /* Collection of all color transform proposals */
  def defaultColorProposal(implicit rnd: Random):
  ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {
    val colorC = GaussianColorProposal(RGB(0.01f, 0.01f, 0.01f), 0.01f, RGB(1e-4f, 1e-4f, 1e-4f))
    val colorF = GaussianColorProposal(RGB(0.001f, 0.001f, 0.001f), 0.01f, RGB(1e-4f, 1e-4f, 1e-4f))
    val colorHF = GaussianColorProposal(RGB(0.0005f, 0.0005f, 0.0005f), 0.01f, RGB(1e-4f, 1e-4f, 1e-4f))

    MixtureProposal(0.2f *: colorC + 0.6f *: colorF + 0.2f *: colorHF).toParameterProposal
  }


  def fit(targetFn: String, lmFn: String, outputDir: String, modelRenderer: AlbedoMoMoRenderer, expression: Boolean = true, gamma: Boolean = true)(implicit rnd: Random): RenderParameter = {

    val g = 1.0 / 2.2 // gamma

    def applyGamma(i: PixelImage[RGBA]): PixelImage[RGBA] = i.map(c => RGBA(pow(c.r, g), pow(c.g, g), pow(c.b, g)))

    def inverseGamma(i: PixelImage[RGBA]): PixelImage[RGBA] = i.map(c => RGBA(pow(c.r, 1.0 / g), pow(c.g, 1.0 / g), pow(c.b, 1.0 / g)))

    val loadimage = PixelImageIO.read[RGBA](new File(targetFn)).get
    val target = if (gamma)
      inverseGamma(loadimage)
    else
      loadimage

    val targetLM = TLMSLandmarksIO.read2D(new File(lmFn)).get.filter(lm => lm.visible)

    PixelImageIO.write(target, new File(s"$outputDir/target.png")).get

    if (gamma)
      PixelImageIO.write(applyGamma(target), new File(s"$outputDir/target_Gamma.png")).get


    val init: RenderParameter = RenderParameter.defaultSquare.fitToImageSize(target.width, target.height)


    val sdev = 0.043f

    /* Foreground Evaluator */
    val pixEval = IsotropicGaussianPixelEvaluator(sdev)

    /* Background Evaluator */
    val histBGEval = HistogramRGB.fromImageRGBA(target, 25)

    /* Pixel Evaluator */
    val imgEval = IndependentPixelEvaluator(pixEval, histBGEval)

    /* Prior Evaluator */
    val priorEval = ProductEvaluator(GaussianShapePrior(0, 1), GaussianTexturePrior(0, 1))

    /* Image Evaluator */
    val allEval = ImageRendererEvaluator(modelRenderer, imgEval.toDistributionEvaluator(target))

    /* Landmarks Evaluator */
    val pointEval = IsotropicGaussianPointEvaluator[_2D](4.0) //lm click uncertainty in pixel! -> should be related to image/face size
    val landmarksEval = LandmarkPointEvaluator(targetLM, pointEval, modelRenderer)


    //logging
    val imageLogger = ImageRenderLogger(modelRenderer, new File(s"$outputDir/"), "mc-").withBackground(target)

    // Metropolis logger
    val printLogger = PrintLogger[RenderParameter](Console.out, "").verbose
    val mhLogger = printLogger


    // keep track of best sample
    val bestFileLogger = ParametersFileBestLogger(allEval, new File(s"$outputDir/fit-best.rps"))
    val bestSampleLogger = BestSampleLogger(allEval)
    val parametersLogger = ParametersFileLogger(new File(s"$outputDir/"), "mc-")

    val fitLogger = bestFileLogger :+ bestSampleLogger

    // pose proposal
    val totalPose = defaultPoseProposal(modelRenderer)

    //light proposals
    val lightProposal = defaultIlluminationProposalNoOptimizer

    //color proposals
    val colorProposal = defaultColorProposal

    //Morphable Model  proposals
    val momoProposal = if (expression) defaultMorphableModelProposal else neutralMorphableModelProposal


    // full proposal filtered by the landmark and prior Evaluator
    val proposal = MetropolisFilterProposal(MetropolisFilterProposal(MixtureProposal(totalPose + colorProposal + 3f *: momoProposal + 2f *: lightProposal), landmarksEval), priorEval)

    //pose and image chains
    val imageFitter = MetropolisHastings(proposal, allEval)
    val poseFitter = MetropolisHastings(totalPose, landmarksEval)


    println("everyting setup. starting fitter ...")


    //landmark chain for initialisation
    val initDefault: RenderParameter = RenderParameter.defaultSquare.fitToImageSize(target.width, target.height)
    val init10 = initDefault.withMoMo(init.momo.withNumberOfCoefficients(50, 50, 5))
    val initLMSamples: IndexedSeq[RenderParameter] = poseFitter.iterator(init10, mhLogger).take(5000).toIndexedSeq

    val lmScores = initLMSamples.map(rps => (landmarksEval.logValue(rps), rps))

    val bestLM = lmScores.maxBy(_._1)._2
    RenderParameterIO.write(bestLM, new File(s"$outputDir/fitter-lminit.rps")).get

    val imgLM = modelRenderer.renderImage(bestLM)
    PixelImageIO.write(imgLM, new File(s"$outputDir/fitter-lminit.png")).get

    def printer(sample: RenderParameter): RenderParameter = {
      println(s"${sample.momo.shape} ${sample.momo.color} ${sample.momo.expression}")
      sample
    }

    // image chain, fitting
    val fitsamples = imageFitter.iterator(bestLM, mhLogger).loggedWith(fitLogger).take(10000).toIndexedSeq
    val best = bestSampleLogger.currentBestSample().get

    val imgBest = modelRenderer.renderImage(best)
    PixelImageIO.write(imgBest, new File(s"$outputDir/fitter-best.png")).get

    if (gamma)
      PixelImageIO.write(applyGamma(imgBest), new File(s"$outputDir/fitter-best_Gamma.png")).get
    best

    val imgBestDiffuse = modelRenderer.renderImageDiffuse(best)
    PixelImageIO.write(imgBestDiffuse, new File(s"$outputDir/fitter-best-diffuse.png")).get

    val imgBestSpecular = modelRenderer.renderImageSpecular(best)
    PixelImageIO.write(imgBestSpecular, new File(s"$outputDir/fitter-best-specular.png")).get
    best
  }
}