// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import sfa.SFAWordsTest;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertNotNull;


@RunWith(JUnit4.class)
public abstract class AbstractClassifierTest {

    private static final double MAX_ERROR_VALUES_PER_SAMPLING_SET = 5;
    protected static final File DATASETS_DIRECTORY = new File(AbstractClassifierTest.class.getClassLoader().getResource("datasets/").getFile());

    @Test
    public void testClassificationOnUCRData() {
        // the relative path to the datasetsArray
        ClassLoader classLoader = SFAWordsTest.class.getClassLoader();
        for (DataSet dataSet : getDataSets()) {
            Classifier classifier = trainClassifier(dataSet);
            assertNotNull(classifier);
        }
    }

    @Test
    public void testSave() throws IOException {
        DataSet dataSet = this.getDataSets().get(0);
        testSaveLoadGivesEqualTestResults(dataSet);
    }

    private void testSaveLoadGivesEqualTestResults(DataSet dataSet) throws IOException {
        Classifier classifier=trainClassifier(dataSet);
        File file=createTempClassifierFile();
        classifier.save(file);
        Classifier loadedClassifier = Classifier.load(file);
        Assert.assertNotNull(loadedClassifier);
        checkEqualsResultsOfClassifiers(dataSet, classifier, loadedClassifier);
        file.deleteOnExit();
    }

    private void checkEqualsResultsOfClassifiers(DataSet dataSet, Classifier classifier, Classifier loadedClassifier) {
        TimeSeries[] samples = TimeSeriesLoader.loadDataset(getFirstTrainFile(dataSet));
        Classifier.Predictions loadedScore = loadedClassifier.score(samples);
        Classifier.Predictions score = classifier.score(samples);

        Assert.assertArrayEquals(loadedScore.labels, score.labels);
        Assert.assertEquals(loadedScore.correct.get(), score.correct.get());
    }

    private File getFirstTrainFile(DataSet dataset) {
        return getTrainFiles(dataset)[0];
    }

    private File createTempClassifierFile() throws IOException {
        return File.createTempFile("classifier", "class");

    }

    protected static final class DataSet{
        public DataSet(String name, double trainingAccuracy, double testingAccuracy) {
            this.name = name;
            this.trainingAccuracy = trainingAccuracy;
            this.testingAccuracy = testingAccuracy;
        }

        String name; double trainingAccuracy, testingAccuracy;
    }

    protected Classifier trainClassifier(DataSet dataSet) {
        File[] trainFiles = getTrainFiles(dataSet);

        Classifier classifier = null;
        for (File train : trainFiles) {
            File test = new File(train.getAbsolutePath().replaceFirst("TRAIN", "TEST"));

            if (!test.exists()) {
                System.err.println("File " + test.getName() + " does not exist");
                test = null;
            }

            Classifier.DEBUG = false;

            // Load the train/test splits
            TimeSeries[] testSamples = TimeSeriesLoader.loadDataset(test);
            TimeSeries[] trainSamples = TimeSeriesLoader.loadDataset(train);

            classifier = initClassifier();
            Classifier.Score scoreW = classifier.eval(trainSamples, testSamples);
            assertEquals("testing result of " +
                    dataSet.name+" does NOT match",
                    scoreW.getTestingAccuracy(),
                    dataSet.testingAccuracy,
                    calcDelta(scoreW.testSize));
            assertEquals("training result of "+dataSet.name+" does NOT match",
                    scoreW.getTrainingAccuracy(),
                    dataSet.trainingAccuracy,
                    calcDelta(scoreW.trainSize));
            System.out.println(scoreW.toString());

        }
        return classifier;
    }

    protected File[] getTrainFiles(DataSet dataSet) {
        File dataSetDirectory = new File(DATASETS_DIRECTORY.getAbsolutePath()+"/"+dataSet.name);
        return getTrainFilesFromDir(dataSetDirectory);
    }

    private File[] getTrainFilesFromDir(File dataSetDirectory) {
        return dataSetDirectory.listFiles(pathname -> pathname.getName().toUpperCase().endsWith("TRAIN"));
    }

    /**
     * calculates the delta value to allow a maximum of error values per training or test cases
     * @param samplingSize
     * @return
     */
    private double calcDelta(int samplingSize) {
        return MAX_ERROR_VALUES_PER_SAMPLING_SET /(double)samplingSize;
    }

    protected abstract List<DataSet> getDataSets();


    protected abstract Classifier initClassifier();
}
