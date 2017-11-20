name := "dataAnalysis"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies ++=Seq("org.apache.spark" %% "spark-core" % "2.0.1","org.apache.spark" %% "spark-sql" % "2.0.1","org.apache.spark" %% "spark-graphx" % "2.0.1","org.apache.spark" % "spark-streaming_2.10" % "2.0.1","org.apache.spark" %% "spark-mllib" % "2.0.1")
