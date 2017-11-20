import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.hadoop.io._
import org.apache.spark._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}
import scala.collection.immutable.ListMap
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.{SQLContext, Row, DataFrame}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.Pipeline 
import org.apache.spark.ml.classification.DecisionTreeClassificationModel 
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator 
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler} 
import org.apache.spark.ml.clustering.KMeans

object dataAnalysis {
    //class with all the values equal to the features we retain
    case class privacy(AdvertisingNetworks:Double,PrivacyofCommunications:Double,KeyEscrowEncryption:Double,ChildrensPrivacy:Double,AttitudesTowardsSpamming:Double, InternetPrivacyLaws:Double)

    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("Assignment5")
        val sc = new SparkContext(conf)
        //creating SQL context
        val SQLCtx = new org.apache.spark.sql.SQLContext(sc)
        import SQLCtx.implicits._
        val sparkSession = SparkSession.builder.getOrCreate()
        import sparkSession.implicits._
        //reading input from file
        val textFile=sc.textFile(args(0))
        val data=textFile.collect
        //removing headers and converting data from categorical to numeric
        val dataWithoutHeader=data.filter(x=>(!x.contains("Advertising Networks"))).map(x => x.split("\\s{2,3}|\t"))
        val numericData=dataWithoutHeader.map(x=>x.map(_.replace("Agree Somewhat","1").replace("Agree Strongly","2").replace("Disagree Somewhat","3").replace("Disagree Strongly","4").replace("Disagree Strongly ","5").replace("Neither Agree or Disagree","6").replace("No Opinion","7").replace("Delete","1").replace("Not App","2").replace("Other","3").replace("Read","4").replace("Reply","5").replace("Retaliate","6")))
        val transformedData=numericData.map(x=>privacy(x(0).toString.trim.toFloat,x(22).toString.trim.toFloat,x(8).toString.trim.toFloat,x(3).toString.trim.toFloat,x(2).toString.trim.toFloat,x(7).toString.trim.toFloat))
        //creating a dataframe from the RDD
        val schemaString="AdvertisingNetworks PrivacyofCommunications KeyEscrowEncryption ChildrensPrivacy AttitudesTowardsSpamming InternetPrivacyLaws"
        val fields=schemaString.split(" ").map(fieldName => StructField(fieldName, StringType, nullable = false))
        val schema = StructType(fields)
        val rows=transformedData.map(x=>Row(x.AdvertisingNetworks.toString,x.PrivacyofCommunications.toString,x.KeyEscrowEncryption.toString,x.ChildrensPrivacy.toString,x.AttitudesTowardsSpamming.toString,x.InternetPrivacyLaws.toString))
      val rowsrdd=sc.parallelize(rows)
       val DDF=SQLCtx.createDataFrame(rowsrdd, schema)
       //casting dataframe from string to double
       val newDF=DDF.withColumn("AdvertisingNetworks",DDF("AdvertisingNetworks").cast(DoubleType)).withColumn("PrivacyofCommunications",DDF("PrivacyofCommunications").cast(DoubleType)).withColumn("KeyEscrowEncryption",DDF("KeyEscrowEncryption").cast(DoubleType)).withColumn("ChildrensPrivacy",DDF("ChildrensPrivacy").cast(DoubleType)).withColumn("AttitudesTowardsSpamming",DDF("AttitudesTowardsSpamming").cast(DoubleType)).withColumn("InternetPrivacyLaws",DDF("InternetPrivacyLaws").cast(DoubleType))
       val privacyFeatures= newDF.columns.drop(1)
       val assembledFeatures = new VectorAssembler().setInputCols(privacyFeatures).setOutputCol("features")
       val trainingData = assembledFeatures.transform(newDF)
       //creating model
       val myLinearRegObj = new LinearRegression().setMaxIter(1000).setRegParam(0.1).setElasticNetParam(0.4).setFeaturesCol("features").setLabelCol("AdvertisingNetworks").setPredictionCol("predicted_AdvertisingNetworks")
       val myLinearRegModel = myLinearRegObj.fit(trainingData)
       //printing coefficients
       val trainingSummary = myLinearRegModel.summary
       val testData = trainingData.limit(1)
       //predicting model with test data
       val PredictedDF = myLinearRegModel.transform(testData)
       //end of regression classification begins
       //splitting the data into test and train 
       val Array(trainingdata, testdata) = trainingData.randomSplit(Array(0.7, 0.3))
       //selecting features
       val decissionTreeClassifier = new DecisionTreeClassifier().setLabelCol("AdvertisingNetworks").setFeaturesCol("features")
       //training
       val model = decissionTreeClassifier.fit(trainingdata)
       //testing
       val predictions = model.transform(testdata)
       val evaluator = new MulticlassClassificationEvaluator().setLabelCol("AdvertisingNetworks").setPredictionCol("prediction").setMetricName("accuracy")
       val accuracy = evaluator.evaluate(predictions)
       //end of classification below is the code for K means
       val kmeans = new KMeans().setK(7).setSeed(1L) 
       //taking k as 7 and training the making clusters. 7 beacuse it is optimim number of clusters from R
       val model2 = kmeans.fit(trainingData)
       val WSSSE = model2.computeCost(trainingData)
       println()
       println("*********************regression***************************************")
       println(s"Coefficients: ${myLinearRegModel.coefficients} Intercept: ${myLinearRegModel.intercept}")
       println(s"numIterations: ${trainingSummary.totalIterations}")
       println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
       println(s"r2: ${trainingSummary.r2}")
       PredictedDF.show()
       println("*********************classification***************************************")
       println("accuracy = " + accuracy)
       println("Test Error = " + (1.0 - accuracy))
       predictions.show()
       model2.clusterCenters.foreach(println)
       println("*********************clustering***************************************")
       println("WSSE = " + WSSSE)
       println("cluster Centers:")
       println(model2.clusterCenters.foreach(println))
       println("************************************************************")
    }
}
