/*
 * Copyright 2007-2010 
 * This file is part of ALVS.
 *
 * ALVS is free software; you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 *
 * ALVS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * ALVS; if not, write to the Free Software Foundation, Inc., 51 Franklin St,
 * Fifth Floor, Boston, MA 02110-1301 USA
 */
package alvs.util.internalframe;

import java.awt.Dimension;
import javax.swing.JInternalFrame;
import javax.swing.JScrollPane;
import javax.swing.JTable;

/**
 * Internal frame which will contain a table with the data set.
 * 
 * @author scsandra
 */
public class DataInternalFrame extends JInternalFrame {

        JTable table;

        public DataInternalFrame(String name, JTable table, Dimension size) {
                super(name, true, true, true, true);
                this.table = table;
                setSize(size);
                setTable(table);
        }

        public JTable getTable() {
                return table;
        }

        public void setTable(JTable table) {
                try {
                        JScrollPane scrollPanel = new JScrollPane(table);
                        scrollPanel.setPreferredSize(new Dimension(this.getWidth() - 330, this.getHeight() - 90));
                        this.add(scrollPanel);
                } catch (Exception e) {
                        e.printStackTrace();
                }
        }
}
