package sfa.classification.weaselCharacterClassifier;

import java.util.ArrayList;
import java.util.List;

import sfa.classification.AbstractClassifierTest;
import sfa.classification.Classifier;
import sfa.classification.WEASELCharacterClassifier;
import subwordTransformer.apriori.AprioriParameter;
import subwordTransformer.apriori.AprioriTransformer;

public class AprioriClassifierTest extends AbstractClassifierTest {
  @Override
  protected List<DataSet> getDataSets() {
    List<DataSet> dataSets = new ArrayList<>();

    dataSets.add(new DataSet("variable_length", 0.929, 0.964));
    dataSets.add(new DataSet("Coffee", 1.0, 1.0));
    dataSets.add(new DataSet("Beef", 0.933, 0.833));
    dataSets.add(new DataSet("CBF", 1.0, 0.987));

    return dataSets;
  }

  @Override
  protected Classifier initClassifier() {
    WEASELCharacterClassifier.transformer = new AprioriTransformer(WEASELCharacterClassifier.maxS);
    WEASELCharacterClassifier.transformerParameterList = new ArrayList<>(AprioriParameter.getParameterList(2, 2, 0.5, 0.5, 1));
    return new WEASELCharacterClassifier();
  }

}
