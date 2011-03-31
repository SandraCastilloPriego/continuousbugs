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
package GopiBugs.modules.configuration.parameters;

import GopiBugs.data.BugDataset;
import GopiBugs.data.ParameterSet;
import GopiBugs.desktop.Desktop;
import GopiBugs.desktop.GopiBugsMenu;
import GopiBugs.main.GopiBugsCore;
import GopiBugs.main.GopiBugsModule;
import GopiBugs.taskcontrol.Task;
import GopiBugs.taskcontrol.TaskListener;
import GopiBugs.taskcontrol.TaskStatus;
import GopiBugs.util.GUIUtils;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.util.logging.Logger;

/**
 *
 * @author scsandra
 */
public class ParameterConfiguration implements GopiBugsModule, TaskListener, ActionListener {

        private Logger logger = Logger.getLogger(this.getClass().getName());
        private Desktop desktop;
        final String helpID = GUIUtils.generateHelpID(this);

        public void initModule() {

                this.desktop = GopiBugsCore.getDesktop();
                desktop.addMenuItem(GopiBugsMenu.CONFIGURATION, "Parameters Configuration..",
                        "Parameters configuration", KeyEvent.VK_P, this, null, null);
        }

        public void taskStarted(Task task) {
                logger.info("Parameters configuration");
        }

        public void taskFinished(Task task) {
                if (task.getStatus() == TaskStatus.FINISHED) {
                        logger.info("Finished Parameters configuration ");
                }

                if (task.getStatus() == TaskStatus.ERROR) {

                        String msg = "Error while Parameters configuration  .. ";
                        logger.severe(msg);
                        desktop.displayErrorMessage(msg);

                }
        }

        public void actionPerformed(ActionEvent e) {
                BugDataset[] dataset = desktop.getSelectedDataFiles();
                if (dataset.length > 0) {
                        ParameterDialog dialog = new ParameterDialog("Data parameters configuration", helpID, dataset[0]);
                        dialog.setVisible(true);
                }
        }

        public ParameterSet getParameterSet() {
                return null;
        }

        public void setParameters(ParameterSet parameterValues) {
        }

        public String toString() {
                return "Parameters configuration";
        }
}
