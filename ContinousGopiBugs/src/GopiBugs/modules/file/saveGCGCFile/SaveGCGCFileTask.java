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
package GopiBugs.modules.file.saveGCGCFile;

import GopiBugs.data.BugDataset;
import GopiBugs.data.DatasetType;
import GopiBugs.data.impl.SimpleParameterSet;
import GopiBugs.data.writer.WriteFile;
import GopiBugs.taskcontrol.Task;
import GopiBugs.taskcontrol.TaskStatus;

/**
 *
 * @author scsandra
 */
public class SaveGCGCFileTask implements Task {

    private BugDataset dataset;
    private TaskStatus status = TaskStatus.WAITING;
    private String errorMessage;
    private String path;
    private WriteFile writer;
    private SimpleParameterSet parameters;

    public SaveGCGCFileTask(BugDataset dataset, SimpleParameterSet parameters, String path) {
        this.dataset = dataset;
        this.path = path;
        this.parameters = parameters;
        writer = new WriteFile();
    }

    public String getTaskDescription() {
        return "Saving Dataset... ";
    }

    public double getFinishedPercentage() {
        return 0.0;
    }

    public TaskStatus getStatus() {
        return status;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public void cancel() {
        status = TaskStatus.CANCELED;
    }

    public void run() {
        try {
            status = TaskStatus.PROCESSING;
            if (dataset.getType() == DatasetType.GCGCTOF) {
                if (parameters.getParameterValue(SaveGCGCParameters.type).toString().matches(".*Excel.*")) {
                    writer.WriteExcelFileGCGC(dataset, path, parameters);
                } else {
                    writer.WriteCommaSeparatedFileGCGC(dataset, path, parameters);
                }
            }
            status = TaskStatus.FINISHED;
        } catch (Exception e) {
            status = TaskStatus.ERROR;
        }
    }
}
