package sfa.classification;

import java.util.ArrayList;
import java.util.List;

public class ShotgunEnsembleClassifierTest extends AbstractClassifierTest{
    @Override
    protected List<DataSet> getDataSets() {
        List<DataSet> dataSets=new ArrayList<>();
        //variable_length;Shotgun Ensemble;0.929;0.857
        dataSets.add(new DataSet("variable_length", 0.929, 0.857));
        //Coffee;Shotgun Ensemble;1.0;0.929
        dataSets.add(new DataSet("Coffee", 1.0, 0.929));
        //Beef;Shotgun Ensemble;0.667;0.9
        dataSets.add(new DataSet("Beef", 0.667,0.9));
        //CBF;Shotgun Ensemble;0.967;0.991
        dataSets.add(new DataSet("CBF", 0.967, 0.991));
        return dataSets;
    }

    @Override
    protected Classifier initClassifier() {
        return new ShotgunEnsembleClassifier();
    }
}
