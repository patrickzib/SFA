package sfa.classification;

import java.util.ArrayList;
import java.util.List;

public class ShotgunClassifierTest extends AbstractClassifierTest{
    @Override
    protected List<DataSet> getDataSets() {
        List<DataSet> dataSets=new ArrayList<>();
        //Coffee;BOSS Ensemble;1.0;1.0
        dataSets.add(new DataSet("Coffee", 1.0, 1.0));
        //Beef;BOSS Ensemble;1.0;0.733
        dataSets.add(new DataSet("Beef", 0.633, 0.8));
        //CBF;BOSS Ensemble;1.0;0.999
        dataSets.add(new DataSet("CBF", 1.0, 0.969));
        return dataSets;
    }

    @Override
    protected Classifier initClassifier() {
        return new ShotgunClassifier();
    }
}
