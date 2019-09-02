package sfa.classification.weaselCharacterClassifier;

import java.util.ArrayList;
import java.util.List;

import sfa.classification.AbstractClassifierTest;
import sfa.classification.Classifier;
import sfa.classification.WEASELCharacterClassifier;
import subwordTransformer.apriori.AprioriParameter;
import subwordTransformer.apriori.SupervisedAprioriTransformer;

public class SupervisedAprioriClassifierTest extends AbstractClassifierTest {
  @Override
  protected List<DataSet> getDataSets() {
    List<DataSet> dataSets = new ArrayList<>();

    dataSets.add(new DataSet("variable_length", 0.929, 0.964));
    dataSets.add(new DataSet("Coffee", 1.0, 1.0));
    dataSets.add(new DataSet("Beef", 0.9, 0.833));
    dataSets.add(new DataSet("CBF", 0.967, 0.997));

    return dataSets;
  }

  @Override
  protected Classifier initClassifier() {
    WEASELCharacterClassifier.transformer = new SupervisedAprioriTransformer(WEASELCharacterClassifier.maxS);
    WEASELCharacterClassifier.transformerParameterList = new ArrayList<>(AprioriParameter.getParameterList(2, 4, 0.5, 0.5, 1));
    return new WEASELCharacterClassifier();
  }

}
