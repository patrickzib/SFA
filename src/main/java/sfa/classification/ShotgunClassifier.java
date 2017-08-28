// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import sfa.timeseries.TimeSeries;

/**
 *  The Shotgun Classifier as published in:
 *
 *    Schäfer, P.: Towards time series classification without human preprocessing.
 *    In Machine Learning and Data Mining in Pattern Recognition,
 *    pages 228–242. Springer, 2014.
 *
 * @author bzcschae
 *
 */
public class ShotgunClassifier extends Classifier {

    public ShotgunClassifier(TimeSeries[] train, TimeSeries[] test) throws IOException {
        super(train, test);
    }

    public Score eval() throws IOException {
        ExecutorService exec = Executors.newFixedThreadPool(threads);
        try {

            // Shotgun Distance
            Score totalBestScore = null;
            int bestCorrectTesting = 0;
            int bestCorrectTraining = 0;

            for (boolean normMean : NORMALIZATION) {
                long startTime = System.currentTimeMillis();

                this.correctTraining = new AtomicInteger(0);

                List<Score> scores = fitEnsemble(exec, this.trainSamples, normMean, 1.0);

                // training score
                Score bestScore = scores.get(0);
                if (DEBUG) {
                    System.out.println("Shotgun Training:\t" + bestScore.windowLength + "\tnormed: \t" + normMean);
                    outputResult(this.correctTraining.get(), startTime, this.trainSamples.length);
                }

                // Classify: testing score
                int correctTesting = predict(bestScore.windowLength, normMean, this.testSamples, this.trainSamples, 1.0).correct.get();

                if (bestCorrectTraining < bestScore.training) {
                    bestCorrectTesting = correctTesting;
                    bestCorrectTraining = this.correctTraining.get();
                    totalBestScore = bestScore;
                }
                if (DEBUG) {
                    System.out.println("");
                }
            }
            return new Score(
                       "Shotgun",
                       1-formatError(bestCorrectTesting, this.testSamples.length),
                       1-formatError(bestCorrectTraining, this.trainSamples.length),
                       totalBestScore.normed,
                       totalBestScore.windowLength);
        }
        finally {
            exec.shutdown();
        }
    }


    public List<Score> fitEnsemble(
        final ExecutorService exec,
        final TimeSeries[] trainSamples,
        final boolean normMean,
        final double factor) {
        int minWindowLength = 5;
        int maxWindowLength = MAX_WINDOW_LENGTH;
        for (TimeSeries ts : trainSamples) {
            maxWindowLength = Math.min(ts.getLength(), maxWindowLength);
        }

        ArrayList<Integer> windows = new ArrayList<Integer>();
        for (int windowLength = maxWindowLength; windowLength >= minWindowLength; windowLength--) {
            windows.add(windowLength);
        }

        return fit(windows.toArray(new Integer[] {}), normMean, trainSamples, factor, exec);
    }

    public List<Score> fit(
        final Integer[] allWindows,
        final boolean normMean,
        final TimeSeries[] samples,
        final double factor,
        ExecutorService exec) {
        final List<Score> results = new ArrayList<Score>(allWindows.length);
        ParallelFor.withIndex(exec, threads, new ParallelFor.Each() {
            Score bestScore = new Score("Shotgun", 0, 0, normMean, 0);
            @Override
            public void run(int id, AtomicInteger processed) {
                for (int i = 0; i < allWindows.length; i++) {
                    if (i % threads == id) {
                        Predictions p = predict(
                                            allWindows[i],
                                            normMean,
                                            samples,
                                            samples,
                                            1.0
                                        );

                        Score score = new Score("Shotgun", p.correct.get(), p.correct.get(), normMean, allWindows[i]);

                        // keep best scores
                        synchronized(this.bestScore) {
                            if (this.bestScore.compareTo(score)<=0) {
                                ShotgunClassifier.this.correctTraining.set((int)score.training);
                                this.bestScore = score;
                            }
                        }

                        // add to ensemble
                        if (score.training >= ShotgunClassifier.this.correctTraining.get() * factor) { // all with same score
                            synchronized(results) {
                                results.add(score);
                            }
                        }
                    }
                }
            }
        });

        // sort descending
        Collections.sort(results, Collections.reverseOrder());
        return results;
    }

    public Predictions predict(
        final int windowLength,
        final boolean normMean,
        final TimeSeries[] testSamples,
        final TimeSeries[] trainSamples,
        final double factor) {

        final Predictions p = new Predictions(new String[testSamples.length], 0);

        // calculate means and stds for each sample
        final double[][] means = new double[trainSamples.length][];
        final double[][] stds = new double[trainSamples.length][];
        calcMeansStds(windowLength, trainSamples, means, stds, normMean);

        ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
            @Override
            public void run(int id, AtomicInteger processed) {
                // iterate each sample to classify
                for (int i = 0; i < testSamples.length; i++) {
                    if (i % BLOCKS == id) {
                        final TimeSeries query = testSamples[i];

                        double distanceTo1NN = Double.MAX_VALUE;
                        String predictedLabel = "";

                        int wQueryLen = Math.min(query.getLength(), windowLength);
                        TimeSeries[] disjointWindows = query.getDisjointSequences(wQueryLen, normMean); // possible without copying!?

                        // perform a 1-NN search
                        for (int j = 0; j < trainSamples.length; j++) {
                            TimeSeries ts = trainSamples[j];

                            // Shotgun Distance
                            if (ts != query) {
                                double totalDistance = 0.0;

                                for (final TimeSeries q : disjointWindows) {
                                    double resultDistance = distanceTo1NN;

                                    // calculate euclidean distances for each sliding window
                                    for (int ww = 0, end = ts.getLength()-windowLength+1; ww < end; ww++) { // faster than reevaluation in for loop
                                        double distance = getEuclideanDistance(ts, q, means[j][ww], stds[j][ww], resultDistance, ww);
                                        resultDistance = Math.min(distance,resultDistance);
                                    }
                                    totalDistance += resultDistance;

                                    // pruning on distance
                                    if (totalDistance > distanceTo1NN) {
                                        break;
                                    }
                                }

                                // choose minimum
                                if (totalDistance < distanceTo1NN) {
                                    predictedLabel = ts.getLabel();
                                    distanceTo1NN = totalDistance;
                                }
                            }
                        }

                        // check if the prediction is correct
                        p.labels[i] = predictedLabel;
                        if (testSamples[i].getLabel().equals(p.labels[i])) {
                            p.correct.incrementAndGet();
                        }
                    }
                }
            }
        });

        return p;
    }

    public static double getEuclideanDistance(
        TimeSeries ts,
        TimeSeries q,
        double meanTs,
        double stdTs,
        double minValue,
        int w
    ) {

        double distance = 0.0;
        double[] tsData = ts.getData();
        double[] qData = q.getData();

        for (int ww = 0; ww < qData.length; ww++) {
            double value1 = (tsData[w+ww]-meanTs) * stdTs;
            double value = qData[ww] - value1;
            distance += value*value;

            // early abandoning
            if (distance >= minValue) {
                return Double.MAX_VALUE;
            }
        }

        return distance;
    }

    public static void calcMeansStds(
        final int windowLength,
        final TimeSeries[] trainSamples,
        final double[][] means,
        final double[][] stds,
        boolean normMean) {
        for (int i = 0; i < trainSamples.length; i++) {
            int w = Math.min(windowLength, trainSamples[i].getLength());
            means[i] = new double[trainSamples[i].getLength()-w+1];
            stds[i] = new double[trainSamples[i].getLength()-w+1];
            TimeSeries.calcIncreamentalMeanStddev(w, trainSamples[i].getData(), means[i], stds[i]);
            for (int j = 0; j < stds[i].length; j++) {
                stds[i][j] = (stds[i][j]>0? 1.0 / stds[i][j] : 1.0);
                means[i][j] = normMean? means[i][j] : 0;
            }

        }
    }
}
