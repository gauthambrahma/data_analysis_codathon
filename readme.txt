To compile the project go to the project folder and type "sbt package" without quotes

To run the code use the following command with output path as the path of the text file along with file name. 
Before the command is run please make sure that the current directory is same as the one containing the jar file which is "Project/target/scala-2.10/" after sbt package is run. or from the JAR file folder

spark-submit --class "dataAnalysis" dataanalysis_2.10-1.0.jar <file path with file name>
Example: spark-submit --class "dataAnalysis" dataanalysis_2.10-1.0.jar /home/ponnaga/DAproj/final_privacy.txt
 
for R code use the R studio by copy pasting the lines in the console 