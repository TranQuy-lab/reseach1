"""Generate a Vietnamese report and figures from completed pilot artifacts.

Run from repository root. Never use this script to choose a model or epoch.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAMES = {'edge_mlp':'MLP đặc trưng cạnh', 'sage':'E-GraphSAGE baseline',
         'sage_edge':'E-GraphSAGE + cạnh', 'random_forest':'Random Forest'}


def fmt(mean, sd):
    return f'{mean:.4f} ± {sd:.4f}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='research/artifacts/pilot_data')
    p.add_argument('--runs', default='research/artifacts/pilot_runs')
    p.add_argument('--output', default='research/results')
    p.add_argument('--report', default='research/REPORT_VI.md')
    a = p.parse_args()
    data, runs, output = Path(a.data), Path(a.runs), Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    table = pd.read_csv(runs/'runs.csv')
    if len(table) != 12 or table.groupby('model').seed.nunique().to_dict() != {k:3 for k in NAMES}:
        raise ValueError('Expected all 12 prespecified runs')
    audit = json.loads((data/'audit.json').read_text())
    splits = json.loads((data/'splits.json').read_text())
    provenance = json.loads((runs/'provenance.json').read_text())
    pre = json.loads((runs/'preprocessor.json').read_text())
    classes = pre['classes']
    values, per_class = {}, []
    (output/'metrics').mkdir()
    (output/'histories').mkdir()
    for model in NAMES:
        records = []
        for seed in (11,22,33):
            path = runs/f'{model}_seed{seed}'
            rec = json.loads((path/'metrics.json').read_text())
            records.append(rec)
            shutil.copyfile(path/'metrics.json',output/'metrics'/f'{model}_seed{seed}.json')
            shutil.copyfile(path/'history.json',output/'histories'/f'{model}_seed{seed}.json')
        values[model] = records
        for c in classes:
            scores = [r['test']['per_class'][c]['f1-score'] for r in records]
            per_class.append(dict(model=model,attack=c,support=int(records[0]['test']['per_class'][c]['support']),
                                  mean_f1=float(np.mean(scores)),sd_f1=float(np.std(scores,ddof=1))))
    per_class = pd.DataFrame(per_class)
    per_class.to_csv(output/'per_class.csv',index=False)
    table.to_csv(output/'runs.csv',index=False)
    summary = []
    for model in NAMES:
        part = table.loc[table.model == model]
        row = {'model':model}
        for col in ('test_macro_f1','test_weighted_f1','test_accuracy','seconds_fit_and_evaluate'):
            row[col+'_mean'] = float(part[col].mean())
            row[col+'_sd'] = float(part[col].std())
        summary.append(row)
    summary = pd.DataFrame(summary).set_index('model')
    summary.to_csv(output/'summary.csv')
    leader = summary.test_macro_f1_mean.idxmax()
    stability = ('ít' if summary.loc['sage_edge','test_macro_f1_sd'] < summary.loc['sage','test_macro_f1_sd'] else 'nhiều')
    for src,name in [(data/'audit.json','audit.json'), (data/'splits.json','splits.json'),
                     (runs/'provenance.json','provenance.json'),
                     (Path('research/artifacts/graph_diagnostics.json'),'graph_diagnostics.json')]:
        shutil.copyfile(src,output/name)

    # Raw seed points and mean±SD; bounded metric axis; no significance stars.
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'}):
        fig,axes = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        colors = ['#005A8D','#B24B00','#00684B','#6A3D80']
        labels = ['Edge MLP','E-GraphSAGE','E-GraphSAGE + edge','Random Forest']
        for ax,col,title in zip(axes,['test_macro_f1','test_weighted_f1'],['Macro-F1','Weighted-F1']):
            for i,(model,color) in enumerate(zip(NAMES,colors)):
                scores = table.loc[table.model == model,col].to_numpy()
                ax.scatter(scores, i+np.array([-.09,0,.09]),color=color,marker='o',s=35,zorder=3)
                ax.errorbar(scores.mean(),i,xerr=scores.std(ddof=1),fmt='D',color='black',capsize=4,markersize=5)
            ax.set(yticks=range(4),yticklabels=labels,xlim=(0,1),xlabel=title)
            ax.invert_yaxis()
            ax.grid(axis='x',alpha=.2)
        fig.suptitle('NF-ToN-IoT-v2: offline pilot, 10,000 held-out flows\nDots: 3 model seeds on one split; diamond/error bar: mean ± SD',fontsize=12)
        for ext in ('png','svg'):
            fig.savefig(output/f'pilot_comparison.{ext}',dpi=180,facecolor='white')
        plt.close(fig)
        matrix = np.array([[per_class.loc[(per_class.model==m)&(per_class.attack==c),'mean_f1'].iloc[0]
                            for c in classes] for m in NAMES])
        fig,ax = plt.subplots(figsize=(12,4.5),layout='constrained')
        im = ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto')
        supports = [splits['splits']['test']['classes'][c] for c in classes]
        ax.set(xticks=range(len(classes)),xticklabels=[f'{c}\nn={n}' for c,n in zip(classes,supports)],
               yticks=range(4),yticklabels=labels)
        for i in range(4):
            for j in range(len(classes)):
                ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white' if matrix[i,j]<.45 else 'black')
        fig.colorbar(im,ax=ax,label='Mean test F1 across 3 model seeds')
        ax.set_title('Per-class F1: small support limits interpretation\nOne held-out split; ransomware has only 2 test flows',pad=12)
        for ext in ('png','svg'):
            fig.savefig(output/f'per_class_f1.{ext}',dpi=180,facecolor='white')
        plt.close(fig)

    metric_rows = []
    for model,row in summary.iterrows():
        metric_rows.append('| '+NAMES[model]+' | '+ ' | '.join(fmt(row[m+'_mean'],row[m+'_sd'])
            for m in ('test_macro_f1','test_weighted_f1','test_accuracy'))+' |')
    support_rows = ['| '+c+' | '+' | '.join(str(splits['splits'][s]['classes'][c]) for s in ('train','val','test'))+' |'
                    for c in classes]
    epoch_rows = []
    for _,r in table[table.model != 'random_forest'].iterrows():
        epoch_rows.append(f'| {r.model} | {int(r.seed)} | {int(r.best_epoch)} | {int(r.epochs_ran)} |')
    replay_error = max(r['replay_max_abs_error'] for records in values.values() for r in records)
    graph = json.loads((output/'graph_diagnostics.json').read_text())
    feature_invalid = sum(v['nonfinite_rows'] for v in audit['invalid_rows'].values())
    source_paths = [str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()]
    hashes = {rel:hashlib.sha256((output/rel).read_bytes()).hexdigest() for rel in source_paths}
    (output/'figure_provenance.json').write_text(json.dumps(dict(
        matplotlib=importlib.metadata.version('matplotlib'), estimator='arithmetic mean',
        uncertainty='sample SD, ddof=1, n=3 model seeds on one split; not confidence intervals',
        input='runs.csv and metrics/', png_dpi=180, inches=[12,4.5],
        purpose='general research report; no journal compliance claimed',sha256=hashes),indent=2))
    text = f'''# Báo cáo khởi động lại nghiên cứu E-GraphSAGE

Ngày: 2026-09-19. Kết quả dưới đây được sinh từ 12 lượt thực nghiệm mới,
không sao chép các điểm số cũ trong repo. Đây là **pilot ngoại tuyến trên
50.000 flow**, chưa phải kết quả toàn quy mô hay bằng chứng về tính mới.

## 1. Những gì đã hoàn thành

- Kiểm toán toàn bộ {audit['total_rows']:,} dòng Parquet GNN đang có.
- Chốt [protocol](PROTOCOL.md) trước khi xem điểm số; lưu checksum protocol,
  source code, dữ liệu và phiên bản thư viện trong provenance.
- Triển khai module độc lập `src/nids_research`, không chạy các script cũ.
- Bốn mô hình × ba seed 11/22/33, cùng split và cùng thông tin đầu vào hợp lệ.
- Lưu scaler, nhãn, cấu hình, trọng số, lịch sử và dự đoán theo ID flow.
- Nạp lại tất cả 12 artifact và dự đoán không cần nhãn: sai lệch xác suất
  lớn nhất {replay_error:.3g}. Biên bản kiểm thử được lưu riêng trong results.

## 2. Kiểm toán dữ liệu: quan sát, không suy đoán

Train cũ: 11.858.347; validation cũ: 1.694.048. Quét cả 5 file, 39 feature
và hai endpoint. Số dòng NaN/Inf ở feature: {feature_invalid}.
Không phát hiện endpoint/nhãn không hợp lệ theo các kiểm tra đã triển khai
(thiếu endpoint/port ngoài khoảng và quan hệ Label–Attack).
Chưa xác nhận tất cả IP bằng parser IP hoặc tính đúng ngữ nghĩa từng feature.

Dấu vân tay gồm feature + endpoint, không gồm nhãn:
- Bản ghi trùng dư: {audit['duplicates']['extra_duplicate_rows']}.
- Nhóm chung train/validation cũ: {audit['duplicates']['groups_shared_by_legacy_train_val']}.
- Nhóm nhãn mâu thuẫn: {audit['duplicates']['conflicting_label_groups_all']}.

Đây là kiểm tra băm hai khóa 64 bit trên dữ liệu đã xử lý, không phải chứng
minh toán học không va chạm hoặc kiểm toán CSV gốc. Không phát hiện trùng
toàn dòng không loại trừ các flow gần giống, cùng host, cùng phiên hoặc cùng
đợt tấn công xuất hiện ở hai tập. Giá trị thiếu đã bị điền 0 trước đó cũng
không thể khôi phục từ Parquet.

Mẫu 50.000 được chọn bằng ưu tiên băm từ toàn bộ train cũ, không lấy riêng
đầu file và không cân bằng lớp. Một đại diện/nhóm được chọn; trong dữ liệu
hiện tại không tìm thấy trùng nên bước đó không loại thêm dòng. Chia mới
35.000/5.000/10.000, không trùng flow ID. Validation cũ không dùng chọn mô hình.

| Lớp | Train | Validation | Test |
|---|---:|---:|---:|
{chr(10).join(support_rows)}

**Ransomware chỉ có 2 mẫu test, mitm 5, Backdoor 11.** Chỉ một dự đoán thay
đổi có thể làm metric lớp hiếm đổi mạnh. Không dùng pilot này để xác nhận
cải thiện lớp hiếm, dù số điểm có cao. Test mới là holdout nội bộ từ train
cũ; không gọi là bộ dữ liệu ngoài chưa từng được dự án biết đến.

## 3. Kết quả chính

Mean ± SD trên **ba seed mô hình**, cùng một split. SD không phải khoảng
tin cậy hiệu quả triển khai. Các số F1 dưới đây ở thang 0–1.

| Mô hình | Macro-F1 test | Weighted-F1 test | Accuracy test |
|---|---:|---:|---:|
{chr(10).join(metric_rows)}

Trong đúng pilot này, {NAMES[leader]} có macro-F1 trung bình cao nhất.
Sage_edge dao động {stability} hơn sage giữa các seed. Chưa có cơ sở xem
thứ hạng pilot là ưu thế tổng quát của một họ mô hình. Không dùng phép
so sánh này để loại bỏ GNN ở mọi quy mô hoặc mọi protocol, nhất là khi
budget và capacity khác nhau.

![Tất cả điểm của ba seed và mean ± SD của bốn mô hình](results/pilot_comparison.png)

Dữ liệu bảng: [runs.csv](results/runs.csv), [summary.csv](results/summary.csv).
Kết quả từng lớp: [per_class.csv](results/per_class.csv); mọi lớp và mọi seed
đều được giữ lại, không chỉ báo lượt tốt nhất.

![F1 trung bình từng lớp với số mẫu test hiển thị](results/per_class_f1.png)

## 4. Cách đọc và giới hạn kết quả

- Chênh lệch giữa các mô hình chỉ là quan sát trong pilot đã định nghĩa.
  Chưa có kiểm định xác nhận ngoài mẫu hoặc nhiều split độc lập.
- Sage và sage_edge dùng cùng encoder/siêu tham số; sage_edge chỉ thêm
  đặc trưng cạnh vào linear head. So sánh này kiểm tra đường đặc trưng
  trực tiếp, không phải toàn bộ biến thể MLP head của repo cũ.
- MLP có 6.410 tham số, sage 88.586, sage_edge 88.976; không cùng capacity.
  Random Forest dùng 100 cây cố định, không tune. Không suy ra ưu thế
  bản chất của một họ mô hình từ ngân sách này.
- Neural models full-batch: 1 epoch = 1 bước cập nhật. Tối đa 120 bước,
  patience=20; có thể chưa hội tụ. Giữ nguyên các lượt early-stop sớm,
  không loại seed kém hoặc kéo dài chỉ một mô hình sau khi xem test.

| Mô hình | Seed | Epoch tốt nhất theo val | Epoch đã chạy |
|---|---:|---:|---:|
{chr(10).join(epoch_rows)}

Đồ thị train có {graph['train']['nodes']:,} đỉnh, khoảng
{100*graph['train']['degree_one_node_fraction']:.1f}% có bậc endpoint bằng 1.
Khoảng {100*graph['test']['flows_with_any_unseen_endpoint']:.1f}% flow test có ít nhất
một endpoint IP:port chưa thấy trong train. Endpoint mới không đồng nghĩa
host/network mới: đổi port cũng tạo endpoint mới. Mẫu ngẫu nhiên làm thay
đổi topology, nên chưa ngoại suy lợi ích sang graph đầy đủ.

Đầu phân loại chỉ dựa vào cặp đỉnh có giới hạn với nhiều flow cùng endpoint,
nhưng chẩn đoán hậu nghiệm ở test này chỉ thấy
{graph['test']['conflicting_label_endpoint_pairs']} cặp endpoint chứa nhiều nhãn.
Do đó không được coi giới hạn đó là nguyên nhân đã chứng minh của mọi sai số.

## 5. Đối chiếu bài báo và sửa diễn giải cũ

- Đây là baseline thích nghi v2, không tái lập nguyên trạng E-GraphSAGE.
  Bảo toàn endpoint thay vì random IP từng dòng; không target-encode;
  split 70/10/20; eval mode; checkpoint theo val; chấm một lần/flow.
- Bài gốc NF-ToN-IoT khác v2; weighted-F1 0,63 đa lớp khác macro-F1 và
  binary-F1. Không dùng các con số đó làm phép đối chứng trực tiếp.
- Bảng V của PDF v8 đã có DoS/XSS F1=0; không gọi thất bại lớp hiếm là
  khám phá mới chỉ dựa vào kết quả cũ.
- Script biểu đồ cũ chuẩn hóa lại validation. Các số cũ phải được đánh
  giá lại bằng đúng scaler train trước khi dùng cho bài viết.

Xem [ghi chú nguồn](EVIDENCE_NOTES_VI.md) cho nguồn đã kiểm tra và phạm vi đọc.

## 6. Hướng nghiên cứu tiếp theo đã chốt

**Ưu tiên độ tin cậy của so sánh trước khi thêm kiến trúc mới.**

1. Chạy nghiên cứu hội tụ/early-stopping trên validation của một protocol
   phát triển riêng, cùng ngân sách cho neural baselines. Không dùng test
   pilot đã xem để xác nhận các lựa chọn mới.
2. Có raw CSV hoặc dữ liệu chứa timestamp/nhóm phiên đáng tin cậy để
   kiểm chứng bước tiền xử lý trước Parquet và thiết kế holdout phù hợp.
3. Tăng cỡ dữ liệu để có đủ lớp hiếm; báo số mẫu và khoảng bất định phù
   hợp cấu trúc phụ thuộc. Nhiều seed không thay thế nhiều mẫu hiếm.
4. Kiểm tra độ nhạy một chiều/hai chiều, feature encoding, graph context
   và sampling bằng ablation thay từng yếu tố; thêm MLP gần capacity nếu
   muốn quy lợi ích cho topology chứ không chỉ năng lực mô hình.
5. Chạy trên dữ liệu/mạng khác với nhãn và protocol tương thích trước khi
   tuyên bố tổng quát. Đọc đầy đủ nghiên cứu liên quan trước khi chốt tính mới.

**Trạng thái:** nền kỹ thuật và vòng pilot đã hoàn tất; nghiên cứu toàn quy
mô, tính mới và khả năng triển khai chưa được xác nhận.

## 7. Tái lập và skill

Xem [RUNBOOK_VI.md](RUNBOOK_VI.md) cho lệnh chuẩn bị, huấn luyện và dùng model.
Các artifact lớn ở `research/artifacts/` được bỏ qua khi git add; báo cáo,
metric và biểu đồ nhỏ ở `research/results/` có thể kiểm tra/commit.

Đã áp dụng nckh: scientific-critical-thinking, experimental-design,
scikit-learn, exploratory-data-analysis và scientific-visualization.
Đã ghi nhận nguồn hỗ trợ quy trình: Kassis, T., Agarwal, V., He, Y., Patel,
D., & Brueckner, A. M. (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).
Nguồn này không cung cấp bằng chứng hiệu năng NIDS của pilot.
'''
    report_path = Path(a.report)
    result_rel = os.path.relpath(output, report_path.parent)
    text = text.replace('(results/', f'({result_rel}/')
    report_path.write_text(text)
    print('Wrote',report_path,'and compact evidence under',output)


if __name__ == '__main__':
    main()
