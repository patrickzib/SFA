package sfa.classification;

import java.util.ArrayList;
import java.util.List;

public class BossEnsembleClassifierTest extends AbstractClassifierTest{
    @Override
    protected List<DataSet> getDataSets() {
        List<DataSet> dataSets=new ArrayList<>();

        //variable_length;BOSS Ensemble;1.0;0.964
        dataSets.add(new DataSet("variable_length", 1.0, 0.964));
        //Coffee;BOSS Ensemble;1.0;1.0
        dataSets.add(new DataSet("Coffee", 1.0, 1.0));
        //Beef;BOSS Ensemble;1.0;0.833
        dataSets.add(new DataSet("Beef", 1.0, 0.833));
        //CBF;BOSS Ensemble;1.0;0.999
        dataSets.add(new DataSet("CBF", 1.0, 0.999));
        return dataSets;
    }

    @Override
    protected Classifier initClassifier() {
        return new BOSSEnsembleClassifier();
    }
}
