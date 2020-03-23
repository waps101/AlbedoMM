package faces.lib

import scalismo.faces.parameters.{Camera, ColorTransform, DirectionalLight, MoMoInstance, Pose, RenderParameter, SphericalHarmonicsLight}
import scalismo.faces.sampling.evaluators.LogNormalDistribution
import scalismo.faces.sampling.face.proposals.ParameterProposals
import scalismo.faces.sampling.face.proposals.ParameterProposals.implicits.PartialToFullParameterConverter
import scalismo.sampling.{ProposalGenerator, SymmetricTransitionRatio, TransitionProbability}
import scalismo.utils.Random


/** proposal to change the scaling of the image through the focal length of the camera
 *
 * @param logSdev log of factor of typical variation, 0.0 is no variation */
case class GaussianSpecularProposal(logSdev: Double)(implicit rnd: Random)
  extends ProposalGenerator[DirectionalLight] with TransitionProbability[DirectionalLight] {

  override def propose(current: DirectionalLight):DirectionalLight = {
    val f = math.exp(rnd.scalaRandom.nextGaussian() * logSdev)
    current.copy(shininess = current.shininess * f)
  }

  override def logTransitionProbability(from: DirectionalLight, to: DirectionalLight): Double = {
    if (to.copy(shininess = from.shininess) == from) {
      LogNormalDistribution.logDensity(to.shininess/from.shininess, 0.0, logSdev)
    } else
      Double.NegativeInfinity
  }

  override def toString: String = s"GaussianSpecularProposal($logSdev)"


}




object ParameterProposals {
  /** implicit conversions for promoting partial BetterRenderParameters to full BetterRenderParameter proposals, just use "proposalGenerator.toParameterProposal" */
  object implicits {
    import scala.languageFeature.implicitConversions

    /** implicit wrapper class for partial parameter proposals */
    implicit class NakedPartialParameterProposal[A](proposal: ProposalGenerator[A])(implicit converter: PartialToFullParameterConverter[A]) {
      def toParameterProposal: ProposalGenerator[RenderParameter] = new ProposalGenerator[RenderParameter] {
        override def propose(current: RenderParameter): RenderParameter = {
          val prop = proposal.propose(converter.partialParameter(current))
          converter.fullParameter(prop, current)
        }

        override def toString: String = proposal.toString
      }
    }

    /** implicit wrapper class for partial parameter proposals */
    implicit class PartialParameterProposal[A](proposal: ProposalGenerator[A] with TransitionProbability[A])(implicit converter: PartialToFullParameterConverter[A]) {
      def toParameterProposal: ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = new ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] {
        override def propose(current: RenderParameter): RenderParameter = {
          val prop = proposal.propose(converter.partialParameter(current))
          converter.fullParameter(prop, current)
        }

        override def logTransitionProbability(from: RenderParameter, to: RenderParameter): Double = proposal.logTransitionProbability(converter.partialParameter(from), converter.partialParameter(to))

        override def toString: String = proposal.toString
      }
    }

    /** implicit wrapper class for symmetric partial parameter proposals */
    implicit class PartialSymmetricParameterProposal[A](proposal: ProposalGenerator[A] with SymmetricTransitionRatio[A])(implicit converter: PartialToFullParameterConverter[A]) {
      def toParameterProposal: ProposalGenerator[RenderParameter] with SymmetricTransitionRatio[RenderParameter] = new ProposalGenerator[RenderParameter] with SymmetricTransitionRatio[RenderParameter] {
        override def propose(current: RenderParameter): RenderParameter = {
          val prop = proposal.propose(converter.partialParameter(current))
          converter.fullParameter(prop, current)
        }

        override def toString: String = proposal.toString
      }
    }

    /** implicit wrapper class for symmtric partial parameter proposals with a transition probability */
    implicit class PartialTransitionSymmetricParameterProposal[A](proposal: ProposalGenerator[A] with SymmetricTransitionRatio[A] with TransitionProbability[A])(implicit converter: PartialToFullParameterConverter[A]) {
      def toParameterProposal: ProposalGenerator[RenderParameter] with
        SymmetricTransitionRatio[RenderParameter] with
        TransitionProbability[RenderParameter] =
        new ProposalGenerator[RenderParameter] with
          SymmetricTransitionRatio[RenderParameter] with
          TransitionProbability[RenderParameter] {
          override def propose(current: RenderParameter): RenderParameter = {
            val prop = proposal.propose(converter.partialParameter(current))
            converter.fullParameter(prop, current)
          }

          override def logTransitionProbability(from: RenderParameter, to: RenderParameter): Double = proposal.logTransitionProbability(converter.partialParameter(from), converter.partialParameter(to))

          override def toString: String = proposal.toString
        }
    }

    // all implicits use the same promotion converters

    trait PartialToFullParameterConverter[A] {
      def fullParameter(partial: A, blueprint: RenderParameter): RenderParameter

      def partialParameter(full: RenderParameter): A
    }

    implicit object ParameterAsFullParameter extends PartialToFullParameterConverter[RenderParameter] {
      override def fullParameter(partial: RenderParameter, blueprint: RenderParameter): RenderParameter = partial

      override def partialParameter(full: RenderParameter): RenderParameter = full
    }

    implicit object PoseAsFullParameter extends PartialToFullParameterConverter[Pose] {
      override def fullParameter(partial: Pose, blueprint: RenderParameter): RenderParameter = blueprint.copy(pose = partial)

      override def partialParameter(full: RenderParameter): Pose = full.pose
    }

    implicit object CameraAsFullParameter extends PartialToFullParameterConverter[Camera] {
      override def fullParameter(partial: Camera, blueprint: RenderParameter): RenderParameter = blueprint.copy(camera = partial)

      override def partialParameter(full: RenderParameter): Camera = full.camera
    }

    implicit object ColorAsFullParameter extends PartialToFullParameterConverter[ColorTransform] {
      override def fullParameter(partial: ColorTransform, blueprint: RenderParameter): RenderParameter = blueprint.copy(colorTransform = partial)

      override def partialParameter(full: RenderParameter): ColorTransform = full.colorTransform
    }

    implicit object MoMoInstanceAsFullParameter extends PartialToFullParameterConverter[MoMoInstance] {
      override def fullParameter(partial: MoMoInstance, blueprint: RenderParameter): RenderParameter = {
        val modelPart = blueprint.momo.copy(
          shape = partial.shape,
          color = partial.color,
          expression = partial.expression
        )
        blueprint.copy(momo = modelPart)
      }
      override def partialParameter(full: RenderParameter): MoMoInstance = full.momo
    }

    implicit object SphericalHarmonicsLightAsFullParameter extends PartialToFullParameterConverter[SphericalHarmonicsLight] {
      override def fullParameter(partial: SphericalHarmonicsLight, blueprint: RenderParameter): RenderParameter = blueprint.withEnvironmentMap(partial)

      override def partialParameter(full: RenderParameter): SphericalHarmonicsLight = full.environmentMap
    }

    implicit object DirectionalLightAsFullParameter extends PartialToFullParameterConverter[DirectionalLight] {
      override def fullParameter(partial: DirectionalLight, blueprint: RenderParameter): RenderParameter = blueprint.withDirectionalLight(partial)

      override def partialParameter(full: RenderParameter): DirectionalLight = full.directionalLight
    }
  }
}
