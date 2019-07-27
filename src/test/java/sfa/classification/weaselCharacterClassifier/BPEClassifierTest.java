package sfa.classification.weaselCharacterClassifier;

import java.util.ArrayList;
import java.util.List;

import sfa.classification.AbstractClassifierTest;
import sfa.classification.Classifier;
import sfa.classification.WEASELCharacterClassifier;
import subwordTransformer.bpe.BPEParameter;
import subwordTransformer.bpe.BPETransformer;

public class BPEClassifierTest extends AbstractClassifierTest {
  @Override
  protected List<DataSet> getDataSets() {
    List<DataSet> dataSets = new ArrayList<>();

    dataSets.add(new DataSet("variable_length", 0.929, 0.964));
    dataSets.add(new DataSet("Coffee", 1.0, 1.0));
    dataSets.add(new DataSet("Beef", 0.933, 0.833));
    dataSets.add(new DataSet("CBF", 0.967, 0.996));

    return dataSets;
  }

  @Override
  protected Classifier initClassifier() {
    WEASELCharacterClassifier.transformer = new BPETransformer(WEASELCharacterClassifier.maxS, true);
    WEASELCharacterClassifier.transformerParameterList = new ArrayList<>(BPEParameter.getParameterList(0.5, 0.5, 1));
    return new WEASELCharacterClassifier();
  }

}
