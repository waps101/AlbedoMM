name := """albedoMorphableModel"""
version       := "1.0"

scalaVersion  := "2.11.12"

scalacOptions := Seq("-unchecked", "-deprecation", "-encoding", "utf8")

resolvers += Resolver.jcenterRepo

resolvers += Resolver.bintrayRepo("unibas-gravis", "maven")

libraryDependencies += "ch.unibas.cs.gravis" %% "scalismo-faces" % "0.10.1+"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % "test"

mainClass in assembly := Some("faces.apps.AlbedoModelViewer")

assemblyJarName in assembly := "assembly.jar"
