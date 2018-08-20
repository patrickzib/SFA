package sfa.classification;

import java.util.ArrayList;
import java.util.List;

public class BossVSClassifierTest extends AbstractClassifierTest{
    @Override
    protected List<DataSet> getDataSets() {
        List<DataSet> dataSets=new ArrayList<>();
        //variable_length;BOSS;1.0;1.0
        // dataSets.add(new DataSet("variable_length", 1.0, 1.0)); // FIXME there is a problem with the reproducibility
        //Coffee;BOSS VS;1.0;1.0
        dataSets.add(new DataSet("Coffee", 1.0, 1.0));
        //Beef;BOSS VS;1.0;0.767
        dataSets.add(new DataSet("Beef", 1.0, 0.833));
        //CBF;BOSS VS;1.0;0.998
        dataSets.add(new DataSet("CBF", 1.0, 0.998));
        return dataSets;
    }

    @Override
    protected Classifier initClassifier() {
        return new BOSSVSClassifier();
    }
}
