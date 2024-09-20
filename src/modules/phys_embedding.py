# Physical Embedding Module from the original OCP repository (used in almost every materials discovery model)
# https://github.com/Open-Catalyst-Project/ocp
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn


class PhysEmbedding(nn.Module):
    def __init__(self, props=True, pg=False, short=False) -> None:
        """
        Create physics-aware embeddings meta class with sub-emeddings for each atom

        Args:
            props (bool, optional): Create an embedding of physical
                properties. (default: :obj:`True`)
            pg (bool, optional): Learn two embeddings based on period and
                group information respectively. (default: :obj:`False`)
            short (bool, optional): Remove all columns containing NaN values.
                (default: :obj:`False`)
        """
        super().__init__()

        self.properties_list = [
            "atomic_radius",
            "atomic_volume",
            "density",
            "dipole_polarizability",
            "electron_affinity",
            "en_allen",
            "vdw_radius",
            "metallic_radius",
            "metallic_radius_c12",
            "covalent_radius_pyykko_double",
            "covalent_radius_pyykko_triple",
            "covalent_radius_pyykko",
            "IE1",
            "IE2",
        ]
        self.group_size = 0
        self.period_size = 0
        self.n_properties = 0

        self.props = props
        self.pg = pg
        self.short = short

        group = None
        period = None

        # Dataset was extracted to avoid using mendeleev library
        self.table_path = Path("./data/chemical_elements/elements_table_ionization.csv")
        df = pd.read_csv(self.table_path)

        # Fetch group and period data
        if pg:
            df.group_id = df.group_id.fillna(value=19.0)
            self.group_size = df.group_id.unique().shape[0]
            group = torch.cat(
                [
                    torch.ones(1, dtype=torch.long),
                    torch.tensor(df.group_id.loc[:100].values, dtype=torch.long),
                ]
            )

            self.period_size = df.period.loc[:100].unique().shape[0]
            period = torch.cat(
                [
                    torch.ones(1, dtype=torch.long),
                    torch.tensor(df.period.loc[:100].values, dtype=torch.long),
                ]
            )

        self.register_buffer("group", group) # register buffer is used to store the tensor in the model
        # so that it can be loaded on the GPU without being a parameter
        self.register_buffer("period", period)

        # Create an embedding of physical properties
        if props:
            # Select only potentially relevant elements
            df = df[self.properties_list]
            df = df.loc[:85, :] # Only the first 85 elements are used in the dataset

            # Normalize
            df = (df - df.mean()) / df.std()

            # Process 'NaN' values and remove further non-essential columns
            if self.short:
                self.properties_list = df.columns[~df.isnull().any()].tolist()
                df = df[self.properties_list]
            else:
                self.properties_list = df.columns[
                    pd.isnull(df).sum() < int(1 / 2 * df.shape[0])
                ].tolist()
                df = df[self.properties_list]
                col_missing_val = df.columns[df.isna().any()].tolist()
                df[col_missing_val] = df[col_missing_val].fillna(
                    value=df[col_missing_val].mean()
                )

            self.n_properties = len(df.columns)
            properties = torch.cat(
                [
                    torch.zeros(1, self.n_properties),
                    torch.from_numpy(df.values).float(),
                ]
            )
            self.register_buffer("properties", properties)

    @property
    def device(self):
        if self.props:
            return self.properties.device
        if self.pg:
            return self.group.device
