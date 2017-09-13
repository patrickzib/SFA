package sfa.transformation;

import sfa.classification.Classifier;

import java.util.ArrayList;
import java.util.List;

/**
 * An Ensemble of Classifiers
 */
public class EnsembleModel<E> {

  public List<E> model;
  public List<Double> accuracy;

  public EnsembleModel() {
  }

  /**
   * Create an Ensemble
   *
   * @param models List of models
   * @param accuracies List of accuracies for each model
   */
  public EnsembleModel(List<E> models, List<Double> accuracies) {
    this.model = models;
    this.accuracy = accuracies;
  }

  public E getHighestScoringModel() {
    return model.get(0);
  }

  public double getHighestAccuracy() {
    return accuracy.get(0);
  }
}
