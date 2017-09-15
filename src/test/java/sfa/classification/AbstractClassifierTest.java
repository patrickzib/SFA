// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import java.io.File;
import java.util.List;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import sfa.SFAWordsTest;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;

import static junit.framework.TestCase.assertEquals;


@RunWith(JUnit4.class)
public abstract class AbstractClassifierTest {

    private static final double MAX_ERROR_VALUES_PER_SAMPLING_SET = 5;

    protected static final class DataSet{
        public DataSet(String name, double trainingAccuracy, double testingAccuracy) {
            this.name = name;
            this.trainingAccuracy = trainingAccuracy;
            this.testingAccuracy = testingAccuracy;
        }

        String name; double trainingAccuracy, testingAccuracy;
    }

    // The datasetsArray to use
    protected static String[] datasetsArray = new String[] {"Coffee", "Beef","CBF"};

    @Test
    public void testClassificationOnUCRData() {
        try {
            // the relative path to the datasetsArray
            ClassLoader classLoader = SFAWordsTest.class.getClassLoader();

            File dir = new File(classLoader.getResource("datasets/").getFile());
            //File dir = new File("/Users/bzcschae/workspace/similarity/datasetsArray/classification");

            for (DataSet dataSet : getDataSets()) {
                File dataSetDirectory = new File(dir.getAbsolutePath()+"/"+dataSet.name);

                for (File train : dataSetDirectory.listFiles()) {
                    if (train.getName().toUpperCase().endsWith("TRAIN")) {
                        File test = new File(train.getAbsolutePath().replaceFirst("TRAIN", "TEST"));

                        if (!test.exists()) {
                            System.err.println("File " + test.getName() + " does not exist");
                            test = null;
                        }

                        Classifier.DEBUG = false;

                        // Load the train/test splits
                        TimeSeries[] testSamples = TimeSeriesLoader.loadDataset(test);
                        TimeSeries[] trainSamples = TimeSeriesLoader.loadDataset(train);

                        // The WEASEL-classifier
                        Classifier classifier = initClassifier();
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
                }

            }
        } finally {
            ParallelFor.shutdown();
        }
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
