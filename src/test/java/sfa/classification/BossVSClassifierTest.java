package sfa.classification;

import java.util.ArrayList;
import java.util.List;

public class BossVSClassifierTest extends AbstractClassifierTest{
    @Override
    protected List<DataSet> getDataSets() {
        List<DataSet> dataSets=new ArrayList<>();
        //Coffee;BOSS Ensemble;1.0;1.0
        dataSets.add(new DataSet("Coffee", 1.0, 1.0));
        //Beef;BOSS Ensemble;1.0;0.733
        dataSets.add(new DataSet("Beef", 1.0, 0.833));
        //CBF;BOSS Ensemble;1.0;0.999
        dataSets.add(new DataSet("CBF", 1.0, 0.998));
        return dataSets;
    }

    @Override
    protected Classifier initClassifier() {
        return new BOSSVSClassifier();
    }
}
