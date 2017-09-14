package sfa.classification;

import sfa.classification.Classifier;

import java.util.ArrayList;
import java.util.List;

/**
 * An Ensemble of Classifiers
 */
public class Ensemble<E> {

  public List<E> model;

  public Ensemble() {
  }

  /**
   * Create an Ensemble
   *
   * @param models List of models
   */
  public Ensemble(List<E> models) {
    this.model = models;
  }

  public E getHighestScoringModel() {
    return model.get(0);
  }

  public E get(int i) {
    return model.get(i);
  }

  public int size() {
    return model.size();
  }
}
