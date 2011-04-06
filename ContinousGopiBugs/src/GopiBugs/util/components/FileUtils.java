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
package GopiBugs.util.components;

import GopiBugs.data.BugDataset;
import GopiBugs.data.PeakListRow;
import GopiBugs.data.datamodels.OtherDataModel;
import GopiBugs.data.DatasetType;
import GopiBugs.data.impl.datasets.SimpleBasicDataset;
import GopiBugs.data.impl.peaklists.SimplePeakListRowOther;
import GopiBugs.util.Tables.DataTableModel;

/**
 *
 * @author scsandra
 */
public class FileUtils {

    public static PeakListRow getPeakListRow(DatasetType type) {
        switch (type) {           
            case TRAINING:
            case VALIDATION:
                return new SimplePeakListRowOther();
        }
        return null;
    }

    public static BugDataset getDataset(BugDataset dataset, String Name) {
        BugDataset newDataset = null;
        switch (dataset.getType()) {            
            case TRAINING:
            case VALIDATION:
                newDataset = new SimpleBasicDataset(Name + dataset.getDatasetName());
                break;
        }
        newDataset.setType(dataset.getType());
        return newDataset;
    }

    public static DataTableModel getTableModel(BugDataset dataset) {
        DataTableModel model = null;
        switch (dataset.getType()) {            
            case TRAINING:
            case VALIDATION:
                model = new OtherDataModel(dataset);
                break;
        }
        return model;
    }
}
