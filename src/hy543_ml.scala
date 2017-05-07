import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructType,StructField,StringType}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}

val baseRdd = sc.textFile("/home/mipach/movie_metadata.csv")

case class Movie(Color: String, DirectorName: String, NumCriticReviews: Int, Duration: Int, DirectorFbLikes: Int, Actor3FbLikes: Int, Actor2Name: String, Actor1FbLikes: Int, Gross: Int, Gernes: String, Actor1Name: String, Title: String, NumVotedUsers: Int, CastTotalFbLikes: Int, Actor3Name: String, FacesOnPoster: Int, PlotKeywords: String, IMDBLink: String, NumUserReviews: Int, Language: String, Country: String, ContentRating: String, Budget: Int, Year: Int, Actor2FbLikes: Int, IMDBScore: Double, AspectRatio: Double, MovieFbLikes: Int)

val header = baseRdd.first

val parsedPointsRdd = baseRdd.filter(x => x != header).map(x => x.split(",")).filter(x => x.size == 28).map(p => Movie( if(p(0) != "") p(0) else "NaN",
											 if( p(1) != "") p(1) else "NaN",
											 if(p(2) != "")p(2).toInt else 0,
										 	 if(p(3) != "" ) p(3).toInt else 0,
											 if(p(4) != "" )p(4).toInt else 0,
											 if(p(5) != "" )p(5).toInt else 0,
											 if( p(6) != "")p(6) else "Nan",
											 if(p(7) != "" )p(7).toInt else 0,
											 if(p(8) != "" )p(8).toInt else 0,
											 if(p(9) != "" ) p(9) else "Nan" ,
											 if(p(10) != "") p(10) else "Nan",
											 if(p(11) != "") p(11) else "Nan" ,
											 if(p(12) != "" )p(12).toInt else 0,
											 if(p(13) != "" )p(13).toInt else 0,
											 if(p(14) != "") p(14) else "NaN",
											 if(p(15) != "" ) p(15).toInt else 0,
											 if(p(16) != "") p(16) else "Nan",
											 if(p(17) != "") p(17) else "Nan",
											 if(p(18) != "" )p(18).toInt else 0,
											 if(p(19) != "") p(19) else "Nan" ,
											 if(p(20) != "") p(20) else "Nan",
											 if(p(21) != "") p(21) else "Nan", 
		if(p(22) != "" && BigInt(p(22)) < Int.MaxValue )p(22).toInt else if(p(22) != "") Int.MaxValue else 0,
											 if(p(23) != "" )p(23).toInt else 0,
											 if(p(24) != "" )p(24).toInt else 0,
											 if(p(25) != "" )p(25).toDouble else 0.0,
											 if(p(26) != "" )p(26).toDouble else 0.0,
											 if(p(27) != "" )p(27).toInt else 0))


//for correlation
val df = parsedPointsRdd.toDF

val df2 = df.drop(df.col("Gernes"))
val df3 = df2.drop(df2.col("Title"))
val df4 = df3.drop(df3.col("NumVotedUsers"))
val df5 = df4.drop(df4.col("CastTotalFbLikes"))
val df6 = df5.drop(df5.col("FacesOnPoster"))
val df7 = df6.drop(df6.col("PlotKeywords"))
val df8 = df7.drop(df7.col("IMDBLink"))
val df9 = df8.drop(df8.col("NumUserReviews"))
val df10 = df9.drop(df9.col("Language"))
val df11 = df10.drop(df10.col("Country"))
val df12 = df11.drop(df11.col("ContentRating"))

val colorIndexer = new StringIndexer().setInputCol("Color").setOutputCol("ColorCat")
val directorNameIndexer = new StringIndexer().setInputCol("DirectorName").setOutputCol("DirectorNameCat")
val actor1NameIndexer = new StringIndexer().setInputCol("Actor1Name").setOutputCol("Actor1NameCat")
val actor3NameIndexer = new StringIndexer().setInputCol("Actor3Name").setOutputCol("Actor3NameCat")
val actor2NameIndexer = new StringIndexer().setInputCol("Actor2Name").setOutputCol("Actor2NameCat")

val df13 = colorIndexer.fit(df12).transform(df12)
val df14 = directorNameIndexer.fit(df13).transform(df13)
val df15 = actor1NameIndexer.fit(df14).transform(df14)
val df16 = actor2NameIndexer.fit(df15).transform(df15)
val df17 = actor3NameIndexer.fit(df16).transform(df16)

val df18 = df17.drop(df17.col("Color"))
val df19 = df18.drop(df18.col("DirectorName"))
val df20 = df19.drop(df19.col("Actor2Name"))
val df21 = df20.drop(df20.col("Actor1Name"))
val df22 = df21.drop(df21.col("Actor3Name"))

val rows = new VectorAssembler().setInputCols(df22.columns).setOutputCol("corr_features").transform(df22).select("corr_features").rdd
val items_mllib_vector = rows.map(_.getAs[org.apache.spark.ml.linalg.Vector](0)).map(org.apache.spark.mllib.linalg.Vectors.fromML)
val correlMatrix: Matrix = Statistics.corr(items_mllib_vector, "pearson")

println(correlMatrix.toString(17,Int.MaxValue))
val file = "/home/mipach/correlation.txt"
val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file)))
writer.write(correlMatrix.toString(17,Int.MaxValue))
writer.close

//in vim do :%s/\d\+\.\d\+/\=printf('%.2f',str2float(submatch(0)))/g then :s/\s\+/ /g in every line to reduce the spaces

val Array(trainData, testData) = parsedPointsRdd.randomSplit(Array(.8,.2), 42)

//ML logic
val Array(trainDataDF, testDataDF) = df22.randomSplit(Array(.8,.2), 42)

val sclicer = new VectorSlicer().setInputCol("rawFeatures").setOutputCol("slicedfeatures").setNames(Array("NumCriticReviews","Duration","DirectorFbLikes","Actor3FbLikes","Actor1FbLikes","Gross","Budget","Actor2FbLikes","AspectRatio","MovieFbLikes","ColorCat"))

val scaler = new StandardScaler().setInputCol("slicedfeatures").setOutputCol("features").setWithStd(true).setWithMean(true)

val lr = new LinearRegression().setMaxIter(10).setRegParam(0.0001).setElasticNetParam(0.8).setLabelCol("IMDBScore").setFeaturesCol("features")

val assembler = new VectorAssembler().setInputCols(Array("NumCriticReviews","Duration","DirectorFbLikes","Actor3FbLikes","Actor1FbLikes","Gross","Budget","Actor2FbLikes","AspectRatio","MovieFbLikes","ColorCat")).setOutputCol("rawFeatures")

val lrPipeline = new Pipeline().setStages(Array(assembler,sclicer,scaler,lr))

val lrModel = lrPipeline.fit(trainDataDF)

val lrPredictions = lrModel.transform(testDataDF)

lrPredictions.select("prediction","IMDBScore","features").show(20)

//evaluate

val evaluator = new RegressionEvaluator().setLabelCol("IMDBScore").setPredictionCol("prediction").setMetricName("rmse")

val rmse = evaluator.evaluate(lrPredictions)

//statistics functions
def GreatMoviesStatistics():Double = (parsedPointsRdd.filter(x => x.IMDBScore >= 8.0).count.toDouble / parsedPointsRdd.count)*100

def DirectorStatistics(director:String):Double = (parsedPointsRdd.filter(x => x.DirectorName == director).count.toDouble / parsedPointsRdd.count)*100

def GerneStatistics(gerne:String):Double = (parsedPointsRdd.filter(x => (x.Gernes contains gerne) == true).count.toDouble / parsedPointsRdd.count)*100


//recommendation functions

def BestMoviesInCategory(category:String) = {
      val movies = parsedPointsRdd.filter(x => (x.Gernes.toLowerCase contains category.toLowerCase) == true)
      val sorted = movies.top(10)(Ordering.by[Movie,Double](_.IMDBScore))
      sorted.foreach(x => println(x.Title+" "+x.IMDBScore))
      }

def findTopTen(category:String, language:String, actor1name:String) = { 
	val movies = parsedPointsRdd.filter(x => (x.Gernes.toLowerCase contains category.toLowerCase) == true).filter(x => (x.Language.toLowerCase contains language.toLowerCase) == true).filter(x => (x.Actor1Name.toLowerCase contains actor1name.toLowerCase) == true)
	val sorted = movies.top(10)(Ordering.by[Movie,Double](_.IMDBScore))
	sorted.foreach(x => println(x.Title+" "+x.IMDBScore))
	}


def findTitle(category:String, language:String, actor1name:String, directorname:String) = {
        val movies = parsedPointsRdd.filter(x => (x.Gernes.toLowerCase contains category.toLowerCase) == true).filter(x => (x.Language.toLowerCase contains language.toLowerCase) == true).filter(x => (x.Actor1Name.toLowerCase contains actor1name.toLowerCase) == true).filter(x => (x.DirectorName.toLowerCase contains directorname.toLowerCase) == true)

        val sorted = movies.top(10)(Ordering.by[Movie,Double](_.IMDBScore))
	println("Top possible matches:")
        sorted.foreach(x => println(x.Title))
        }
