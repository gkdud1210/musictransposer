/**
 * VexFlow Handler — 악보 렌더링 모듈
 * ScoreShift AI 악보 조 변환기
 */

const VF = Vex.Flow;

/**
 * 음표 데이터를 VexFlow로 렌더링
 * @param {string}  containerId  - 렌더링할 HTML element ID
 * @param {Array}   notes        - 음표 배열 (pitch, duration, octave, ...)
 * @param {string}  keySignature - 조성 (예: "C", "G", "F")
 * @param {string}  timeSignature - 박자 (예: "4/4", "3/4")
 * @param {object}  options      - 추가 옵션
 */
function renderScore(containerId, notes, keySignature = 'C', timeSignature = '4/4', options = {}) {
  const container = document.getElementById(containerId);
  if (!container) { console.error(`Container #${containerId} not found`); return; }
  container.innerHTML = '';

  const containerWidth = container.offsetWidth || 700;
  const opts = {
    width: containerWidth - 48,
    height: 180,
    staveX: 40,
    staveY: 40,
    clef: 'treble',
    ...options
  };

  const renderer = new VF.Renderer(container, VF.Renderer.Backends.SVG);
  renderer.resize(opts.width, opts.height);
  const context = renderer.getContext();
  context.setFont('Arial', 10);

  // 악보 스타일
  context.setFillStyle('#222');
  context.setStrokeStyle('#222');

  // 여러 마디 계산
  const [beatsPerMeasure] = timeSignature.split('/').map(Number);
  const measures = groupNotesByMeasures(notes, beatsPerMeasure);
  const measuresCount = Math.max(measures.length, 1);

  const firstMeasureWidth = 180;
  const otherMeasureWidth = Math.max(
    140,
    (opts.width - firstMeasureWidth) / Math.max(measuresCount - 1, 1)
  );

  let xOffset = opts.staveX;

  measures.forEach((measureNotes, idx) => {
    const isFirst = idx === 0;
    const staveWidth = isFirst ? firstMeasureWidth : Math.min(otherMeasureWidth, 220);

    const stave = new VF.Stave(xOffset, opts.staveY, staveWidth);

    if (isFirst) {
      stave.addClef(opts.clef);
      stave.addKeySignature(keySignature);
      stave.addTimeSignature(timeSignature);
    }

    stave.setContext(context).draw();

    if (measureNotes.length > 0) {
      const vfNotes = measureNotes.map(n => createVFNote(n));
      const voice = new VF.Voice({
        num_beats: beatsPerMeasure,
        beat_value: parseInt(timeSignature.split('/')[1])
      }).setMode(VF.Voice.Mode.SOFT);

      voice.addTickables(vfNotes);

      const beams = VF.Beam.generateBeams(vfNotes);
      new VF.Formatter().joinVoices([voice]).format([voice], staveWidth - 30);

      voice.draw(context, stave);
      beams.forEach(b => b.setContext(context).draw());
    } else {
      // 빈 마디 → 쉼표
      const rest = new VF.StaveNote({ keys: ['b/4'], duration: 'wr', type: 'r' });
      const voice = new VF.Voice({ num_beats: beatsPerMeasure, beat_value: parseInt(timeSignature.split('/')[1]) })
        .setMode(VF.Voice.Mode.SOFT);
      voice.addTickables([rest]);
      new VF.Formatter().joinVoices([voice]).format([voice], staveWidth - 30);
      voice.draw(context, stave);
    }

    xOffset += staveWidth;

    // 렌더 영역 동적 확장
    if (xOffset > opts.width - 20) {
      renderer.resize(xOffset + 60, opts.height);
    }
  });
}

/**
 * 음표 1개를 VexFlow StaveNote로 변환
 */
function createVFNote(noteData) {
  const { pitch, octave, duration, type, accidental } = noteData;

  let vfKey;
  if (type === 'rest') {
    vfKey = 'b/4';
  } else {
    const pitchLower = (pitch || 'C').toLowerCase().replace('#', '#').replace('b', 'b');
    const oct = octave || 4;
    vfKey = `${pitchLower}/${oct}`;
  }

  const vfDuration = durationToVF(duration);

  const note = new VF.StaveNote({
    keys: [vfKey],
    duration: type === 'rest' ? `${vfDuration}r` : vfDuration,
  });

  // Accidental 추가
  if (accidental === '#') note.addModifier(new VF.Accidental('#'), 0);
  else if (accidental === 'b') note.addModifier(new VF.Accidental('b'), 0);
  else if (accidental === 'n') note.addModifier(new VF.Accidental('n'), 0);

  return note;
}

/**
 * 음표 duration 문자열 → VexFlow duration 코드
 */
function durationToVF(duration) {
  const map = {
    'whole': 'w', 'half': 'h', 'quarter': 'q',
    'eighth': '8', 'sixteenth': '16',
    '1': 'w', '2': 'h', '4': 'q', '8': '8', '16': '16',
    'w': 'w', 'h': 'h', 'q': 'q',
  };
  return map[duration] || 'q';
}

/**
 * 음표 배열을 마디 단위로 그룹핑
 * beat_value에 따른 단순 분할
 */
function groupNotesByMeasures(notes, beatsPerMeasure = 4) {
  if (!notes || notes.length === 0) return [[]];

  const durationBeats = {
    'w': 4, 'whole': 4, '1': 4,
    'h': 2, 'half': 2, '2': 2,
    'q': 1, 'quarter': 1, '4': 1,
    '8': 0.5, 'eighth': 0.5,
    '16': 0.25, 'sixteenth': 0.25,
  };

  const measures = [];
  let currentMeasure = [];
  let currentBeats = 0;

  notes.forEach(note => {
    const beats = durationBeats[note.duration] || 1;
    currentMeasure.push(note);
    currentBeats += beats;

    if (currentBeats >= beatsPerMeasure) {
      measures.push([...currentMeasure]);
      currentMeasure = [];
      currentBeats = 0;
    }
  });

  if (currentMeasure.length > 0) measures.push(currentMeasure);
  return measures;
}

/**
 * 히어로 섹션 데모 악보 렌더링
 */
function renderHeroDemo() {
  const demoContainer = document.getElementById('heroStaff');
  if (!demoContainer) return;

  const demoNotes = [
    { pitch: 'C', octave: 4, duration: 'q' },
    { pitch: 'E', octave: 4, duration: 'q' },
    { pitch: 'G', octave: 4, duration: 'q' },
    { pitch: 'C', octave: 5, duration: 'q' },
    { pitch: 'D', octave: 5, duration: 'h' },
    { pitch: 'B', octave: 4, duration: 'h' },
  ];

  try {
    renderScore('heroStaff', demoNotes, 'C', '4/4', { height: 140 });
  } catch(e) {
    console.warn('Hero demo rendering failed:', e);
  }
}

/**
 * 악보를 SVG로 내보내기
 */
function exportScoreAsSVG(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return null;
  const svg = container.querySelector('svg');
  if (!svg) return null;
  return new XMLSerializer().serializeToString(svg);
}

/**
 * 악보를 PNG로 내보내기 (Canvas 변환)
 */
function exportScoreAsPNG(containerId, callback) {
  const svgStr = exportScoreAsSVG(containerId);
  if (!svgStr) { callback(null); return; }

  const canvas = document.createElement('canvas');
  const img = new Image();
  const svgBlob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(svgBlob);

  img.onload = () => {
    canvas.width = img.width || 700;
    canvas.height = img.height || 200;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    URL.revokeObjectURL(url);
    callback(canvas.toDataURL('image/png'));
  };
  img.src = url;
}

// 페이지 로드 시 히어로 데모 렌더링
window.addEventListener('load', () => {
  setTimeout(renderHeroDemo, 300);
});
