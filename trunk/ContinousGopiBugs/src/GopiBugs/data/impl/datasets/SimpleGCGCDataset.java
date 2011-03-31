/*
 * Copyright 2007-2010 VTT Biotechnology
 * This file is part of GopiBugs.
 *
 * GopiBugs is free software; you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 *
 * GopiBugs is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * GopiBugs; if not, write to the Free Software Foundation, Inc., 51 Franklin St,
 * Fifth Floor, Boston, MA 02110-1301 USA
 */
package GopiBugs.data.impl.datasets;

import GopiBugs.data.impl.peaklists.SimplePeakListRowGCGC;
import GopiBugs.data.impl.*;
import GopiBugs.data.DatasetType;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import GopiBugs.data.BugDataset;
import GopiBugs.data.GCGCColumnName;
import GopiBugs.data.PeakListRow;
import GopiBugs.util.Range;
import java.util.Hashtable;
import java.util.Vector;

/**
 * GCxGC-MS data set implementation.
 *
 * @author scsandra
 */
public class SimpleGCGCDataset implements BugDataset {

    private List<PeakListRow> peakList;
    private Vector<String> nameExperiments;
    private String datasetName;
    private DatasetType type;
    private String infoDataset = "";
    private Hashtable<String, SampleDescription> parameters;
    private Vector<String> parameterNames;
    private int ID;
    private int numberRows = 0;
    Hashtable<String, String> sampleType;

    /**
     *
     * @param names Sample Names
     * @param parameters Alignment parameters
     * @param aligner Class which performed the alignment of the sample files to create this data set
     */
    public SimpleGCGCDataset(String[] names) {

        this.nameExperiments = new Vector<String>();
        for (String experimentName : names) {
            this.nameExperiments.addElement(experimentName);
        }
        // Peak list
        peakList = new ArrayList<PeakListRow>();

        // The data set name is "Alignment" when it is create as a result of the
        // alignment of different sample files
        datasetName = "Alignment";

        this.sampleType = new Hashtable<String, String>();
        // SampleDescription to describe the samples from GopiBugs.modules.configuration.parameters
        this.parameters = new Hashtable<String, SampleDescription>();
        this.parameterNames = new Vector<String>();
    }

    /**
     *
     * @param datasetName Name of data set
     */
    public SimpleGCGCDataset(String datasetName) {
        this.nameExperiments = new Vector<String>();
        peakList = new ArrayList<PeakListRow>();
        this.datasetName = datasetName;
        this.type = DatasetType.GCGCTOF;
        // SampleDescription to describe the samples from GopiBugs.modules.configuration.parameters
        this.parameters = new Hashtable<String, SampleDescription>();
        this.parameterNames = new Vector<String>();
    }

    /**
     * Returns a list of every row in the data set.
     *
     * @return List containing the AlignmentRows in this Alignment
     */
    public List<PeakListRow> getAlignment() {
        return peakList;
    }

    /**
     * Returns a list of rows which contain mass value.
     *
     * @return List of rows
     */
    public List<SimplePeakListRowGCGC> getQuantMassAlignments() {
        List<SimplePeakListRowGCGC> QuantMassList = new ArrayList<SimplePeakListRowGCGC>();
        for (int i = 0; i < peakList.size(); i++) {
            PeakListRow alignmentRow = peakList.get(i);
            if ((Double) alignmentRow.getVar("getMass") > -1) {
                QuantMassList.add((SimplePeakListRowGCGC) alignmentRow);
            }
        }
        return QuantMassList;
    }

    /**
     * Returns an array of a copy of sample names.
     *
     * @return Array with the name of the samples
     */
    public String[] getColumnNames() {
        return (String[]) nameExperiments.toArray(new String[0]).clone();
    }

    /**
     * Adds a new row into the row list inside the class
     *
     * @param row Row
     * @return true when the row was added without problems
     */
    public boolean addAlignmentRow(SimplePeakListRowGCGC row) {
        return peakList.add(row);
    }

    /**
     * Add new rows into the data set. The rows can be in any kind of Collection class.
     *
     * @param rows Rows to be added.
     */
    public void addAll(Collection<? extends SimplePeakListRowGCGC> rows) {
        for (SimplePeakListRowGCGC r : rows) {
            peakList.add(r);
        }
    }

    public void setID(int ID) {
        this.ID = ID;
    }

    public int getID() {
        return ID;
    }

    public void addParameterValue(String experimentName, String parameterName, String parameterValue) {
        if (parameters.containsKey(experimentName)) {
            SampleDescription p = parameters.get(experimentName);
            p.addParameter(parameterName, parameterValue);
        } else {
            SampleDescription p = new SampleDescription();
            p.addParameter(parameterName, parameterValue);
            parameters.put(experimentName, p);
        }
        if (!this.parameterNames.contains(parameterName)) {
            parameterNames.addElement(parameterName);
        }
    }

    public void deleteParameter(String parameterName) {
        for (String experimentName : nameExperiments) {
            if (parameters.containsKey(experimentName)) {
                SampleDescription p = parameters.get(experimentName);
                p.deleteParameter(parameterName);
            }
        }
        this.parameterNames.remove(parameterName);
    }

    public String getParametersValue(String experimentName, String parameterName) {
        if (parameters.containsKey(experimentName)) {
            SampleDescription p = parameters.get(experimentName);
            return p.getParameter(parameterName);
        } else {
            return null;
        }
    }

    public Vector<String> getParameterAvailableValues(String parameter) {
        Vector<String> availableParameterValues = new Vector<String>();
        for (String rawDataFile : this.getAllColumnNames()) {
            String paramValue = this.getParametersValue(rawDataFile, parameter);
            if (!availableParameterValues.contains(paramValue)) {
                availableParameterValues.add(paramValue);
            }
        }
        return availableParameterValues;
    }

    public Vector<String> getParametersName() {
        return parameterNames;
    }

    @Override
    public String toString() {
        return datasetName;
    }

    public String getDatasetName() {
        return datasetName;
    }

    public Vector<String> getAllColumnNames() {
        return nameExperiments;
    }

    public int getNumberCols() {
        return nameExperiments.size();
    }

    public int getNumberRows() {
        return this.peakList.size();
    }

    public int getNumberRowsdb() {
        return this.numberRows;
    }

    public void setNumberRows(int numberRows) {
        this.numberRows = numberRows;
    }

    public void setDatasetName(String name) {
        this.datasetName = name;
    }

    public DatasetType getType() {
        return this.type;
    }

    public void setType(DatasetType type) {
        this.type = type;
    }

    public PeakListRow getRow(int row) {
        return this.peakList.get(row);
    }

    public void removeRow(PeakListRow row) {
        this.peakList.remove(row);
    }

    public void addColumnName(String nameExperiment) {
        this.nameExperiments.add(nameExperiment);
    }

    public void addColumnName(String nameExperiment, int position) {
        this.nameExperiments.insertElementAt(nameExperiment, position);
    }

    public List<PeakListRow> getRows() {
        return this.peakList;
    }

    @Override
    public BugDataset clone() {
        SimpleGCGCDataset newDataset = new SimpleGCGCDataset(datasetName);
        for (String experimentName : this.nameExperiments) {
            newDataset.addColumnName(experimentName);
            for (String parameterName : this.parameterNames) {
                newDataset.addParameterValue(experimentName, parameterName, this.getParametersValue(experimentName, parameterName));
            }
        }
        for (PeakListRow peakListRow : this.peakList) {
            newDataset.addRow(peakListRow.clone());
        }
        newDataset.setType(this.type);

        newDataset.infoDataset = infoDataset;

        return newDataset;

    }

    public void addRow(PeakListRow peakListRow) {
        this.peakList.add(peakListRow);
    }

    public String getInfo() {
        return infoDataset;
    }

    public void setInfo(String info) {
        this.infoDataset = info;
    }

    public PeakListRow[] getRowsInsideRT1AndRT2Range(Range RT1Range, Range RT2Range) {
        List<PeakListRow> rows = new ArrayList<PeakListRow>();
        for (PeakListRow row : this.peakList) {
            if (RT1Range.contains((Double) row.getVar(GCGCColumnName.RT1.getGetFunctionName()))
                    && RT2Range.contains((Double) row.getVar(GCGCColumnName.RT2.getGetFunctionName()))) {
                rows.add(row);
            }
        }
        return rows.toArray(new PeakListRow[0]);
    }

    public PeakListRow[] getRowsInsideRTRange(Range RTRange, int RT) {
        List<PeakListRow> rows = new ArrayList<PeakListRow>();
        for (PeakListRow row : this.peakList) {
            if (RT == 1) {
                if (RTRange.contains((Double) row.getVar(GCGCColumnName.RT1.getGetFunctionName()))) {
                    rows.add(row);
                }
            } else {
                if (RTRange.contains((Double) row.getVar(GCGCColumnName.RT2.getGetFunctionName()))) {
                    rows.add(row);
                }
            }
        }
        return rows.toArray(new PeakListRow[0]);
    }

    public Range getRowsRTRange(int RT) {
        double min = Double.MAX_VALUE;
        double max = 0;
        for (PeakListRow row : this.peakList) {
            if (RT == 1) {
                double RTvalue = (Double) row.getVar(GCGCColumnName.RT1.getGetFunctionName());
                if (RTvalue < min) {
                    min = RTvalue;
                }
                if (RTvalue > max) {
                    max = RTvalue;
                }
            } else {
                double RTvalue = (Double) row.getVar(GCGCColumnName.RT2.getGetFunctionName());
                if (RTvalue < min) {
                    min = RTvalue;
                }
                if (RTvalue > max) {
                    max = RTvalue;
                }
            }
        }
        return new Range(min, max);
    }

    public void setSampleType(String sampleName, String type) {
        this.sampleType.put(sampleName, type);
    }

    public String getSampleType(String sampleName) {
        return sampleType.get(sampleName);
    }
}
