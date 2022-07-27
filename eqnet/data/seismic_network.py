import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
import h5py

def generate_label(phase_list, label_width=[150, 150], nt=8192):

    target = np.zeros([len(phase_list) + 1, nt], dtype=np.float32)

    for i, (picks, w) in enumerate(zip(phase_list, label_width)):
        for phase_time in picks:
            t = np.arange(nt) - phase_time
            gaussian = np.exp(-t**2 / (2 * (w / 6) ** 2))
            gaussian[gaussian < 0.1] = 0.0
            target[i + 1, :] = gaussian
            
    target[0:1, :] = np.maximum(0, 1 - np.sum(target[1:, :], axis=0, keepdims=True))

    return target

class SeismicNetworkIterableDataset(IterableDataset):
    
    degree2km = 111.32
    nt = 8192 ## 8992
    feature_nt = 512 ##560
    feature_scale = int(nt / feature_nt)

    def __init__(self, hdf5_file="ncedc_event.h5", sampling_rate=100.0):
        super().__init__()
        self.hdf5_fp = h5py.File(hdf5_file, "r")
        self.event_ids = list(self.hdf5_fp.keys())
        self.sampling_rate = sampling_rate

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            data_list = self.event_ids
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            data_list = self.event_ids[worker_id::num_workers]
        return iter(self.sample(data_list))

    def __len__(self):
        return len(self.event_ids)

    def sample(self, event_ids):

        idx = np.random.randint(0, len(event_ids))
        event_id = event_ids[idx]
        station_ids = list(self.hdf5_fp[event_id].keys())
        waveforms = np.zeros([3, self.nt, len(station_ids)])
        phase_pick = np.zeros([3, self.nt, len(station_ids)])
        center_heatmap = np.zeros([self.feature_nt, len(station_ids)])
        event_location = np.zeros([4, self.feature_nt, len(station_ids)])
        event_location_mask = np.zeros([self.feature_nt, len(station_ids)])
        relative_position = np.zeros([3, len(station_ids), len(station_ids)])

        for i, sta_id in enumerate(station_ids):

            waveforms[:, :, i] = self.hdf5_fp[event_id+'/'+sta_id][:self.nt,:].T
            attrs = self.hdf5_fp[event_id+'/'+sta_id].attrs
            p_picks = attrs["phase_index"][attrs["phase_type"] == "P"]
            s_picks = attrs["phase_index"][attrs["phase_type"] == "S"]
            phase_pick[:, :, i] = generate_label([p_picks, s_picks], nt=self.nt)

            ## TODO: how to deal with multiple phases
            # center = (self.hdf5_fp[event_id+'/'+sta_id].attrs["phase_index"][::2] + self.hdf5_fp[event_id+'/'+sta_id].attrs["phase_index"][1::2])/2.0
            ## assuming only one event with both P and S picks
            c0 = np.mean(self.hdf5_fp[event_id+'/'+sta_id].attrs["phase_index"]).item()
            dx = round((self.hdf5_fp[event_id].attrs["event_longitude"] - self.hdf5_fp[event_id+'/'+sta_id].attrs["station_longitude"]) * np.cos(np.radians(self.hdf5_fp[event_id].attrs["event_latitude"])) * self.degree2km, 2)
            dy = round((self.hdf5_fp[event_id].attrs["event_latitude"] - self.hdf5_fp[event_id+'/'+sta_id].attrs["station_latitude"]) * self.degree2km, 2)
            dz = round(self.hdf5_fp[event_id].attrs["event_depth_km"] + self.hdf5_fp[event_id+'/'+sta_id].attrs["station_elevation_m"]/1e3, 2)
            # dt = (c0 - self.hdf5_fp[event_id].attrs["event_time_index"]) / self.sampling_rate
            # dt = (c0 - 3000) / self.sampling_rate

            # center_heatmap[int(c0//self.feature_scale), i] = 1
            center_heatmap[:, i] = generate_label([[c0/self.feature_scale],], label_width=[10,], nt=self.feature_nt)[1,:]
            mask = (center_heatmap[:, i] >= 0.5)
            # event_location[0, :, i] = (np.arange(self.nt) - self.hdf5_fp[event_id].attrs["event_time_index"]) / self.sampling_rate
            event_location[0, :, i] = (np.arange(self.feature_nt) - 3000/self.feature_scale) / self.sampling_rate
            event_location[1:, mask, i] = np.array([dx, dy, dz])[:, np.newaxis]
            event_location_mask[:, i] = mask

            for j, sta_jd in enumerate(station_ids):
                relative_position[0, i, j] = round((self.hdf5_fp[event_id+'/'+sta_id].attrs["station_longitude"] - self.hdf5_fp[event_id+'/'+sta_jd].attrs["station_longitude"]) * np.cos(np.radians(self.hdf5_fp[event_id].attrs["event_latitude"])) * self.degree2km, 2)
                relative_position[1, i, j] = round((self.hdf5_fp[event_id+'/'+sta_id].attrs["station_latitude"] - self.hdf5_fp[event_id+'/'+sta_jd].attrs["station_latitude"]) * self.degree2km, 2)
                relative_position[2, i, j] = round((self.hdf5_fp[event_id+'/'+sta_id].attrs["station_elevation_m"] - self.hdf5_fp[event_id+'/'+sta_jd].attrs["station_elevation_m"]) / 1e3, 2)


        yield {
            "waveforms": torch.from_numpy(waveforms).float(),
            "phase_pick": torch.from_numpy(phase_pick).float(),
            "center_heatmap": torch.from_numpy(center_heatmap).float(),
            "event_location": torch.from_numpy(event_location).float(),
            "event_location_mask": torch.from_numpy(event_location_mask).float(),
            "relative_position": torch.from_numpy(relative_position).float()
        }


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    dataset = SeismicNetworkIterableDataset("/Users/weiqiang/Research/EQNet/datasets/NCEDC/ncedc_event.h5")
    for x in dataset:
        # print(x)
        fig, axes = plt.subplots(1, 5, figsize=(15, 5))
        for i in range(x["waveforms"].shape[-1]):

            axes[0].plot((x["waveforms"][-1, :, i])/torch.std(x["waveforms"][-1, :, i])/10 + i)

            axes[1].plot(x["phase_pick"][1, :, i] + i)
            axes[1].plot(x["phase_pick"][2, :, i] + i)

            axes[2].plot(x["center_heatmap"][:, i] + i-0.5)
            # axes[2].scatter(x["event_location"][0, :, i], x["event_location"][1, :, i])

            axes[3].plot(x["event_location"][0, :, i]/10 + i)

            t = np.arange(x["event_location"].shape[1])[x["event_location_mask"][:, i]==1]
            axes[4].plot(t, x["event_location"][1, x["event_location_mask"][:, i]==1, i]/10 + i, color=f"C{i}")
            axes[4].plot(t, x["event_location"][2, x["event_location_mask"][:, i]==1, i]/10 + i, color=f"C{i}")
            axes[4].plot(t, x["event_location"][3, x["event_location_mask"][:, i]==1, i]/10 + i, color=f"C{i}")

        plt.savefig("test.png")
        plt.show()
        
        raise